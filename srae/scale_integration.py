"""Global-scale integration for SRAE hyperparameters (benchmarking variant).

Scale-integrated empirical Bayes (SIEB): a scale mixture over global capacity
multipliers, *conditional on* empirical-Bayes estimates of the relative
per-block scales.

The default SRAE path maximizes the evidence (Type-II / empirical Bayes): a
point estimate of ``(lambda_j, kappa_j, sigma2)``.  Empirical Bayes can be
over-confident about those hyperparameters on small-n data.

This module keeps the same model family (DR spline blocks and optional
interactions), but **integrates** a low-dimensional hyperparameter posterior
and averages predictions. The selected interaction set can still differ from
another variant because the chosen stage-1 fit supplies the screening
residuals:

1. Fit stage-1 hyperparameters ``θ*`` using the Type-II engine by default, or
   the pooled stack when ``base="pooled"``.
2. Freeze *relative* per-block scales at ``θ*`` and put a posterior on global
   multipliers ``(f_λ, f_κ)`` (and ``f_σ`` in regression)::

        λ_j = f_λ · λ*_j ,   κ_j = f_κ · κ*_j

   with prior ``log f ~ N(0, τ²)`` (weakly informative, default ``τ=1``).
3. Sample ``p(f | y) ∝ exp(evidence(f)) · prior(f)`` by random-walk
   Metropolis–Hastings on ``log f``.
4. Predictive distribution = Monte Carlo average of the Gaussian predictive,
   binary moderated-logistic probabilities, or paired-draw multiclass softmax
   probabilities under each sample.

This is **not** full Bayes, and the distinction is deliberate.  Step 2 fixes
the relative per-block scales at data-dependent values ``θ*``, so the prior
on ``(lambda_j, kappa_j)`` is itself estimated from the data -- the defining
empirical-Bayes move.  Only the global multipliers carry a genuine prior and
a genuine posterior.  What is integrated is the **capacity / smoothness
scale**, which is the direction that actually drives overfitting.

Sampling every block precision instead is not intractable -- the log evidence
is analytically differentiable in ``log lambda`` / ``log kappa``, so a
gradient-based sampler handles that dimension routinely.  Two things argue
against it here.  First, the random-walk MH used below mixes poorly at that
dimension, so it would need a different sampler.  Second, and more
substantively, replacing ``θ*`` with a genuine hyperprior reintroduces the
prior sensitivity that empirical Bayes exists to avoid: at the sample sizes
these variants target, the choice of hyperprior scale moves results more than
the sampler does.

The two tasks are also not equally addable.  Gaussian regression is conjugate
and its evidence is exact, so full Bayes over the hyperparameters is a
feasible extension.  The logistic path already relies on a Laplace
approximation to the evidence, so a full-Bayes treatment there would
additionally require sampling ``beta`` itself, with no conjugacy to lean on.

Original ``SRAERegressor`` / ``SRAEClassifier`` / pooled variants are unchanged.

Drop-in::

    from srae import SRAEClassifier, SRAEClassifierSI

    m = SRAEClassifierSI(feature_names=names, interactions=False, n_knots=8)
    m.min_ess_, m.max_rhat_      # check before quoting any uncertainty
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .inference import (
    BlockSpec,
    _assemble_precision,
    _map_logistic,
    _sigmoid,
    fit_gaussian_eb,
    fit_logistic_eb,
    _A_MIN,
    _A_MAX,
)
from .model import SRAEClassifier, SRAERegressor, _InterceptSpec, _BaseSRAE
from .pooled import fit_gaussian_eb_pooled, fit_logistic_eb_pooled

__all__ = [
    "SRAERegressorSI",
    "SRAEClassifierSI",
    "SRAERegressorSIPooled",
    "SRAEClassifierSIPooled",
    "fit_gaussian_si",
    "fit_logistic_si",
]


# ---------------------------------------------------------------------------
# Evidence at fixed hyperparameters
# ---------------------------------------------------------------------------

def _set_scales(blocks, base_lams, base_kaps, f_lam, f_kap):
    for b, lam0, kap0 in zip(blocks, base_lams, base_kaps):
        if isinstance(b, _InterceptSpec):
            continue
        if b.pen.any():
            b.lam = float(np.clip(lam0 * f_lam, _A_MIN, _A_MAX))
        if (~b.pen).any():
            b.kap = float(np.clip(kap0 * f_kap, _A_MIN, _A_MAX))


def _snapshot_base(blocks):
    lams, kaps = [], []
    for b in blocks:
        lams.append(float(b.lam))
        kaps.append(float(b.kap))
    return lams, kaps


def _gaussian_log_evidence(Z, y, blocks, sigma2, M=None, Zty=None):
    """Log marginal likelihood at fixed (blocks, sigma2)."""
    n, p = Z.shape
    if M is None:
        M = Z.T @ Z
    if Zty is None:
        Zty = Z.T @ y
    a = _assemble_precision(blocks)
    H = M / sigma2 + np.diag(a)
    c, low = cho_factor(H, lower=True)
    beta = cho_solve((c, low), Zty / sigma2)
    Sigma = cho_solve((c, low), np.eye(p))
    resid = y - Z @ beta
    rss = float(resid @ resid)
    logdetH = 2.0 * float(np.sum(np.log(np.diag(c))))
    log_ev = (
        -0.5 * n * np.log(2 * np.pi * sigma2)
        - 0.5 * rss / sigma2
        - 0.5 * float(beta @ (a * beta))
        + 0.5 * float(np.sum(np.log(a)))
        - 0.5 * logdetH
    )
    return float(log_ev), beta, Sigma


def _logistic_log_evidence(Z, y, blocks, beta0=None):
    a = _assemble_precision(blocks)
    beta0 = np.zeros(Z.shape[1]) if beta0 is None else beta0
    beta, Sigma, ll, logdetH = _map_logistic(Z, y, a, beta0)
    log_ev = (
        ll
        - 0.5 * float(beta @ (a * beta))
        + 0.5 * float(np.sum(np.log(a)))
        - 0.5 * logdetH
    )
    return float(log_ev), beta, Sigma


def _log_prior_f(log_f, tau):
    """Independent N(0, tau^2) on each log-scale factor."""
    log_f = np.asarray(log_f, float)
    return float(-0.5 * np.sum((log_f / tau) ** 2) - len(log_f) * np.log(tau * np.sqrt(2 * np.pi)))


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

#: Target acceptance rate for the adaptive random-walk proposal.  The 0.234
#: optimum is asymptotic in dimension; here the sampler is 2-3 dimensional, so
#: this is a reasonable operating point rather than a theoretical optimum.
_TARGET_ACCEPT = 0.234

#: Below this effective sample size the Monte Carlo error on a posterior mean
#: exceeds ~1/sqrt(100) = 10% of the posterior SD, which is where reported
#: uncertainty starts being dominated by the sampler rather than the posterior.
_MIN_ESS = 100.0

#: Upper bound on how many draws retain a full ``(beta, Sigma)`` pair.
#:
#: Every draw records its scale factors -- scalars, so ESS and split-Rhat are
#: always computed on the *complete* chain.  Only the coefficient arrays are
#: thinned, because ``Sigma`` is ``q x q``: at the default 8000 draws a model
#: with 310 design columns would otherwise hold 6.2 GB, and one with 780
#: columns about 39 GB.  A thinned subsample of a Markov chain is still a
#: valid Monte Carlo sample for prediction, but it is a *finite* approximation:
#: fewer heavy draws means higher Monte Carlo variance in the predictive
#: average than using every retained draw.  With ESS on the scales around the
#: diagnostic threshold (~100), scale uncertainty is usually the larger error
#: source, so the memory trade-off is intentional rather than free.
_MAX_STORED_DRAWS = 128

#: Split-Rhat above this is flagged as non-convergence.  This is the classical
#: Gelman-Rubin threshold rather than the stricter 1.01 now often recommended:
#: the ``f_lam`` posterior is heavy-tailed (p99 an order of magnitude above the
#: median), so a random-walk sampler needs impractically many draws to push
#: split-Rhat below 1.01 even when the run is perfectly usable.
_MAX_RHAT = 1.05


def _array_draws(samples):
    """Draws that carry ``beta``/``Sigma`` (the thinned predictive subsample)."""
    return [s for s in samples if "beta" in s]


def _ess_1d(x):
    """Effective sample size via Geyer's initial positive sequence.

    Returns ``len(x)`` for an i.i.d. sequence and drops toward 1 as the chain
    becomes sticky.  A constant chain (every proposal rejected) has no
    information, so it returns 1.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return float(n)
    var = x.var()
    if var <= 0:                       # constant chain: zero information
        return 1.0
    c = np.correlate(x - x.mean(), x - x.mean(), mode="full")[n - 1:]
    rho = c / (np.arange(n, 0, -1) * var)
    total = 0.0
    for t in range(1, n - 1, 2):       # sum adjacent pairs while positive
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        total += pair
    denom = 1.0 + 2.0 * total
    return float(np.clip(n / denom, 1.0, n)) if denom > 0 else 1.0


def _ess(chains):
    """Total ESS across chains (list of 1-D sequences)."""
    return float(sum(_ess_1d(c) for c in chains))


def _split_rhat(chains):
    """Split-Rhat (Gelman-Rubin on half-chains).

    Splitting each chain in half makes the statistic sensitive to trends
    *within* a chain, not just disagreement between them.  Needs at least two
    half-chains of length >= 2; returns NaN when that is not available.
    """
    halves = []
    for c in chains:
        c = np.asarray(c, dtype=float)
        h = c.size // 2
        if h >= 2:
            halves.extend([c[:h], c[h:2 * h]])
    if len(halves) < 2:
        return float("nan")
    m, n = len(halves), halves[0].size
    means = np.array([h.mean() for h in halves])
    variances = np.array([h.var(ddof=1) for h in halves])
    W = variances.mean()
    if W <= 0:                         # every half constant
        return float("nan") if np.ptp(means) == 0 else float("inf")
    B = n * means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def _chain_diagnostics(chains_of_samples, keys):
    """ESS and split-Rhat for each scale factor.

    ``chains_of_samples`` is a list (one per chain) of lists of sample dicts.
    """
    ess, rhat = {}, {}
    for k in keys:
        series = [[s[k] for s in chain] for chain in chains_of_samples if chain]
        if not series:
            continue
        ess[k] = _ess(series)
        rhat[k] = _split_rhat(series)
    return ess, rhat


def _warn_if_unconverged(ess, rhat, n_kept):
    """Emit a RuntimeWarning when the run cannot support its own output."""
    if not ess:
        return
    worst_ess = min(ess.values())
    finite = [v for v in rhat.values() if np.isfinite(v)]
    worst_rhat = max(finite) if finite else float("nan")
    problems = []
    if worst_ess < _MIN_ESS:
        problems.append(
            f"effective sample size {worst_ess:.1f} (of {n_kept} draws) is below "
            f"{_MIN_ESS:.0f}; Monte Carlo error is ~{1/np.sqrt(max(worst_ess,1)):.0%} "
            f"of the posterior SD"
        )
    if np.isfinite(worst_rhat) and worst_rhat > _MAX_RHAT:
        problems.append(f"split-Rhat {worst_rhat:.3f} exceeds {_MAX_RHAT}")
    if problems:
        warnings.warn(
            "scale-integration sampler may not have converged: "
            + "; ".join(problems)
            + ". Increase n_samples / n_burn / n_chains, or inspect ess_ and "
              "rhat_ on the fitted estimator.",
            RuntimeWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# MH samplers on global scale factors
# ---------------------------------------------------------------------------

def _valid_f(f_lam, f_kap, f_sig, min_f, max_f, sample_sigma):
    if not (min_f <= f_lam <= max_f and min_f <= f_kap <= max_f):
        return False
    if sample_sigma and not (min_f <= f_sig <= max_f):
        return False
    return True


def _mh_global_gaussian(
    Z, y, blocks, sigma2_map,
    base_lams, base_kaps,
    n_samples=2000, n_burn=1000, thin=1,
    step=0.35, adapt_step=True, tau_prior=1.0,
    log_f_prior_mean=0.0,
    min_f=1.0, max_f=50.0,
    sample_sigma=True, rng=None, verbose=False,
    max_stored=_MAX_STORED_DRAWS,
):
    """Sample (f_lam, f_kap[, f_sig]) with MH; return weighted predictive ingredients.

    ``min_f=1`` (default) truncates the prior on ``log f`` to ``f >= 1``: an
    informative prior encoding that at small ``n`` the Type-II MAP is, if
    anything, under-regularized.  Truncation is a prior choice, not an
    approximation — the posterior is exact under the truncated prior.  Set
    ``min_f`` near 0 for an untruncated two-sided scale posterior.
    """
    rng = np.random.default_rng(rng)
    M = Z.T @ Z
    Zty = Z.T @ y
    min_f = 1e-3 if min_f is None else float(min_f)
    max_f = float(max_f)
    mu = np.full(3 if sample_sigma else 2, float(log_f_prior_mean))

    dim = 3 if sample_sigma else 2
    # start at max(1, exp(prior mean)) so we begin inside the allowed region
    f0 = float(max(min_f, np.exp(log_f_prior_mean)))
    log_f = np.full(dim, np.log(f0))
    f_lam = f_kap = f0
    f_sig = f0 if sample_sigma else 1.0
    _set_scales(blocks, base_lams, base_kaps, f_lam, f_kap)
    sigma2 = float(sigma2_map * f_sig) if sample_sigma else float(sigma2_map)
    log_ev, beta, Sigma = _gaussian_log_evidence(Z, y, blocks, sigma2, M=M, Zty=Zty)
    log_post = log_ev + _log_prior_f(log_f - mu[:dim], tau_prior)

    samples = []
    n_accept = 0
    n_accept_kept = 0
    stride = max(1, int(np.ceil(n_samples / max(int(max_stored), 1))))
    log_step = np.log(max(float(step), 1e-6))
    total = n_burn + n_samples * thin
    for it in range(total):
        accepted = False
        prop = log_f + np.exp(log_step) * rng.standard_normal(dim)
        f_lam_p = float(np.exp(prop[0]))
        f_kap_p = float(np.exp(prop[1]))
        f_sig_p = float(np.exp(prop[2])) if sample_sigma else 1.0
        if _valid_f(f_lam_p, f_kap_p, f_sig_p, min_f, max_f, sample_sigma):
            _set_scales(blocks, base_lams, base_kaps, f_lam_p, f_kap_p)
            sig2_p = float(np.clip(sigma2_map * f_sig_p, 1e-12, None))
            log_ev_p, beta_p, Sigma_p = _gaussian_log_evidence(
                Z, y, blocks, sig2_p, M=M, Zty=Zty
            )
            log_post_p = log_ev_p + _log_prior_f(prop - mu[:dim], tau_prior)
            if np.log(rng.uniform()) < (log_post_p - log_post):
                log_f = prop
                log_ev, beta, Sigma = log_ev_p, beta_p, Sigma_p
                log_post = log_post_p
                f_lam, f_kap, f_sig = f_lam_p, f_kap_p, f_sig_p
                sigma2 = sig2_p
                n_accept += 1
                accepted = True
        # Robbins-Monro adaptation, burn-in only: the proposal is frozen before
        # the first retained draw, so the sampled chain is a plain MH chain
        # with a fixed kernel and needs no diminishing-adaptation argument.
        if adapt_step and it < n_burn:
            log_step += (accepted - _TARGET_ACCEPT) / (it + 1.0) ** 0.6
            log_step = float(np.clip(log_step, np.log(1e-4), np.log(10.0)))
        # always record on schedule (invalid proposals = automatic reject)
        if it >= n_burn and ((it - n_burn) % thin == 0):
            n_accept_kept += accepted
            rec = dict(f_lam=f_lam, f_kap=f_kap, f_sig=f_sig,
                       sigma2=sigma2, log_evidence=log_ev)
            # Coefficient arrays only on a thinned subsample (see
            # _MAX_STORED_DRAWS); the scale factors are always complete.
            if len(samples) % stride == 0:
                rec["beta"] = beta.copy()
                rec["Sigma"] = Sigma.copy()
            samples.append(rec)
        if verbose and (it + 1) % 25 == 0:
            print(f"  MH gauss {it+1}/{total}  accept={n_accept/(it+1):.2f}  "
                  f"step={np.exp(log_step):.3f} "
                  f"f_lam={f_lam:.3f} f_kap={f_kap:.3f}")

    _set_scales(blocks, base_lams, base_kaps, 1.0, 1.0)
    # Report acceptance over the *retained* phase: the burn-in rate is
    # contaminated by adaptation and says nothing about the sampled kernel.
    accept_rate = n_accept_kept / max(len(samples), 1)
    return samples, accept_rate, float(np.exp(log_step))


def _mh_global_logistic(
    Z, y, blocks,
    base_lams, base_kaps,
    n_samples=2000, n_burn=1000, thin=1,
    step=0.35, adapt_step=True, tau_prior=1.0,
    log_f_prior_mean=0.0,
    min_f=1.0, max_f=50.0,
    rng=None, verbose=False, beta0=None,
    max_stored=_MAX_STORED_DRAWS,
):
    rng = np.random.default_rng(rng)
    min_f = 1e-3 if min_f is None else float(min_f)
    max_f = float(max_f)
    mu = np.full(2, float(log_f_prior_mean))
    f0 = float(max(min_f, np.exp(log_f_prior_mean)))
    log_f = np.full(2, np.log(f0))
    f_lam = f_kap = f0
    _set_scales(blocks, base_lams, base_kaps, f_lam, f_kap)
    beta = np.zeros(Z.shape[1]) if beta0 is None else beta0.copy()
    log_ev, beta, Sigma = _logistic_log_evidence(Z, y, blocks, beta0=beta)
    log_post = log_ev + _log_prior_f(log_f - mu, tau_prior)

    samples = []
    n_accept = 0
    n_accept_kept = 0
    stride = max(1, int(np.ceil(n_samples / max(int(max_stored), 1))))
    log_step = np.log(max(float(step), 1e-6))
    total = n_burn + n_samples * thin
    for it in range(total):
        accepted = False
        prop = log_f + np.exp(log_step) * rng.standard_normal(2)
        f_lam_p = float(np.exp(prop[0]))
        f_kap_p = float(np.exp(prop[1]))
        if _valid_f(f_lam_p, f_kap_p, 1.0, min_f, max_f, sample_sigma=False):
            _set_scales(blocks, base_lams, base_kaps, f_lam_p, f_kap_p)
            log_ev_p, beta_p, Sigma_p = _logistic_log_evidence(Z, y, blocks, beta0=beta)
            log_post_p = log_ev_p + _log_prior_f(prop - mu, tau_prior)
            if np.log(rng.uniform()) < (log_post_p - log_post):
                log_f = prop
                log_ev, beta, Sigma = log_ev_p, beta_p, Sigma_p
                log_post = log_post_p
                f_lam, f_kap = f_lam_p, f_kap_p
                n_accept += 1
                accepted = True
        # Burn-in-only adaptation; see the Gaussian sampler for the rationale.
        if adapt_step and it < n_burn:
            log_step += (accepted - _TARGET_ACCEPT) / (it + 1.0) ** 0.6
            log_step = float(np.clip(log_step, np.log(1e-4), np.log(10.0)))
        if it >= n_burn and ((it - n_burn) % thin == 0):
            n_accept_kept += accepted
            rec = dict(f_lam=f_lam, f_kap=f_kap, log_evidence=log_ev)
            if len(samples) % stride == 0:
                rec["beta"] = beta.copy()
                rec["Sigma"] = Sigma.copy()
            samples.append(rec)
        if verbose and (it + 1) % 25 == 0:
            print(f"  MH logit {it+1}/{total}  accept={n_accept/(it+1):.2f}  "
                  f"step={np.exp(log_step):.3f} "
                  f"f_lam={f_lam:.3f} f_kap={f_kap:.3f}")

    _set_scales(blocks, base_lams, base_kaps, 1.0, 1.0)
    accept_rate = n_accept_kept / max(len(samples), 1)
    return samples, accept_rate, float(np.exp(log_step))


# ---------------------------------------------------------------------------
# Fit engines
# ---------------------------------------------------------------------------

def _run_chains(sampler, n_chains, random_state, **kw):
    """Run ``n_chains`` independent MH chains and pool their draws.

    Chains are seeded from one :class:`numpy.random.SeedSequence` so that a
    single ``random_state`` reproduces the whole run, while the chains
    themselves are independent -- which is what makes split-Rhat across them
    meaningful.

    Returns ``(pooled_samples, mean_accept_rate, per_chain_samples, steps)``.
    """
    n_chains = max(int(n_chains), 1)
    seeds = np.random.SeedSequence(random_state).spawn(n_chains)
    per_chain, rates, steps = [], [], []
    for sq in seeds:
        samples, rate, step_used = sampler(rng=np.random.default_rng(sq), **kw)
        per_chain.append(samples)
        rates.append(rate)
        steps.append(step_used)
    pooled = [s for chain in per_chain for s in chain]
    mean_rate = float(np.mean(rates)) if rates else 0.0
    return pooled, mean_rate, per_chain, steps


def _stage1_map(Z, y, blocks, *, task, base, max_iter, tol, verbose, pooled_kwargs):
    """Stage-1 MAP: Type-II or pooled stack. Returns fit dict; mutates blocks."""
    base = (base or "type2").lower()
    if base in ("type2", "eb", "map", "default"):
        if task == "gaussian":
            return fit_gaussian_eb(Z, y, blocks, max_iter=max_iter, tol=tol, verbose=verbose), "type2"
        return fit_logistic_eb(Z, y, blocks, max_iter=max_iter, tol=tol, verbose=verbose), "type2"
    if base in ("pooled", "pool"):
        pk = dict(pooled_kwargs or {})
        pk.setdefault("max_iter", max_iter)
        pk.setdefault("tol", tol)
        pk.setdefault("verbose", verbose)
        if task == "gaussian":
            return fit_gaussian_eb_pooled(Z, y, blocks, **pk), "pooled"
        return fit_logistic_eb_pooled(Z, y, blocks, **pk), "pooled"
    raise ValueError(f"unknown base={base!r}; use 'type2' or 'pooled'")


def fit_gaussian_si(
    Z, y, blocks,
    n_samples=2000, n_burn=1000, thin=1, n_chains=4,
    step=0.35, adapt_step=True, tau_prior=1.0,
    log_f_prior_mean=0.0,
    min_f=1.0, max_f=50.0,
    sample_sigma=True,
    base="type2",
    pooled_kwargs=None,
    max_iter=200, tol=1e-5, verbose=False, random_state=0,
):
    """MAP warm-start (Type-II or pooled), then MH over global (f_λ, f_κ[, f_σ]).

    Parameters
    ----------
    base : {'type2', 'pooled'}
        Hyperparameter MAP used as the unit scale ``f=1``.  ``'pooled'`` runs
        the hierarchical anti-overfitting stack first, then integrates scales
        relative to that (``SRAE*SIPooled``).
    """
    map_fit, base_mode = _stage1_map(
        Z, y, blocks, task="gaussian", base=base,
        max_iter=max_iter, tol=tol, verbose=verbose, pooled_kwargs=pooled_kwargs,
    )
    base_lams, base_kaps = _snapshot_base(blocks)
    sigma2_map = float(map_fit["sigma2"])

    per_chain_stored = max(1, _MAX_STORED_DRAWS // max(int(n_chains), 1))

    def _chain(rng):
        return _mh_global_gaussian(
            Z, y, blocks, sigma2_map, base_lams, base_kaps,
            n_samples=n_samples, n_burn=n_burn, thin=thin,
            step=step, adapt_step=adapt_step, tau_prior=tau_prior,
            log_f_prior_mean=log_f_prior_mean, min_f=min_f, max_f=max_f,
            sample_sigma=sample_sigma,
            rng=rng, verbose=verbose, max_stored=per_chain_stored,
        )

    samples, accept_rate, per_chain, steps = _run_chains(
        _chain, n_chains, random_state
    )
    ess, rhat = _chain_diagnostics(per_chain, ("f_lam", "f_kap", "f_sig"))
    _warn_if_unconverged(ess, rhat, len(samples))
    if not samples:
        samples = [dict(
            f_lam=1.0, f_kap=1.0, f_sig=1.0,
            beta=map_fit["beta"], Sigma=map_fit["Sigma"],
            sigma2=sigma2_map, log_evidence=map_fit["evidence"],
        )]

    heavy = _array_draws(samples)
    betas = np.stack([s["beta"] for s in heavy], axis=0)
    beta_mean = betas.mean(axis=0)
    Sigma_mean = np.mean([s["Sigma"] for s in heavy], axis=0)
    beta_cov = np.cov(betas, rowvar=False) if len(heavy) > 1 else np.zeros_like(Sigma_mean)
    if beta_cov.ndim == 0:
        beta_cov = np.array([[float(beta_cov)]])
    Sigma_mix = Sigma_mean + beta_cov
    sigma2_mean = float(np.mean([s["sigma2"] for s in samples]))

    f_lam_bar = float(np.mean([s["f_lam"] for s in samples]))
    f_kap_bar = float(np.mean([s["f_kap"] for s in samples]))
    _set_scales(blocks, base_lams, base_kaps, f_lam_bar, f_kap_bar)
    a = _assemble_precision(blocks)
    _, beta_f, Sigma_f = _gaussian_log_evidence(Z, y, blocks, sigma2_mean)
    edf = {}
    for b in blocks:
        if isinstance(b, _InterceptSpec):
            continue
        edf[b.name] = b.edf(a[b.sl], np.diag(Sigma_f)[b.sl])

    out = dict(
        beta=beta_mean,
        Sigma=Sigma_mix,
        sigma2=sigma2_mean,
        evidence=float(np.mean([s["log_evidence"] for s in samples])),
        history=map_fit["history"],
        edf=edf,
        n_iter=map_fit["n_iter"],
        samples=samples,
        accept_rate=accept_rate,
        f_lam_mean=f_lam_bar,
        f_kap_mean=f_kap_bar,
        f_sig_mean=float(np.mean([s.get("f_sig", 1.0) for s in samples])),
        map_evidence=map_fit["evidence"],
        base_lams=base_lams,
        base_kaps=base_kaps,
        n_samples=len(samples),
        base_mode=base_mode,
        ess=ess,
        rhat=rhat,
        n_chains=len(per_chain),
        step_adapted=steps,
    )
    # surface useful pooled diagnostics when available
    for k in ("total_edf", "edf_budget", "cal_scale_factor", "n_pruned", "a_floor"):
        if k in map_fit:
            out[f"map_{k}"] = map_fit[k]
    return out


def fit_logistic_si(
    Z, y, blocks,
    n_samples=2000, n_burn=1000, thin=1, n_chains=4,
    step=0.35, adapt_step=True, tau_prior=1.0,
    log_f_prior_mean=0.0,
    min_f=1.0, max_f=50.0,
    base="type2",
    pooled_kwargs=None,
    max_iter=100, tol=1e-5, verbose=False, random_state=0,
):
    """MAP warm-start (Type-II or pooled), then MH over global (f_λ, f_κ)."""
    map_fit, base_mode = _stage1_map(
        Z, y, blocks, task="logistic", base=base,
        max_iter=max_iter, tol=tol, verbose=verbose, pooled_kwargs=pooled_kwargs,
    )
    base_lams, base_kaps = _snapshot_base(blocks)

    per_chain_stored = max(1, _MAX_STORED_DRAWS // max(int(n_chains), 1))

    def _chain(rng):
        return _mh_global_logistic(
            Z, y, blocks, base_lams, base_kaps,
            n_samples=n_samples, n_burn=n_burn, thin=thin,
            step=step, adapt_step=adapt_step, tau_prior=tau_prior,
            log_f_prior_mean=log_f_prior_mean, min_f=min_f, max_f=max_f,
            rng=rng, verbose=verbose, beta0=map_fit["beta"],
            max_stored=per_chain_stored,
        )

    samples, accept_rate, per_chain, steps = _run_chains(
        _chain, n_chains, random_state
    )
    ess, rhat = _chain_diagnostics(per_chain, ("f_lam", "f_kap"))
    _warn_if_unconverged(ess, rhat, len(samples))
    if not samples:
        samples = [dict(
            f_lam=1.0, f_kap=1.0,
            beta=map_fit["beta"], Sigma=map_fit["Sigma"],
            log_evidence=map_fit["evidence"],
        )]

    heavy = _array_draws(samples)
    betas = np.stack([s["beta"] for s in heavy], axis=0)
    beta_mean = betas.mean(axis=0)
    Sigma_mean = np.mean([s["Sigma"] for s in heavy], axis=0)
    beta_cov = np.cov(betas, rowvar=False) if len(heavy) > 1 else np.zeros_like(Sigma_mean)
    if np.ndim(beta_cov) == 0:
        beta_cov = np.array([[float(beta_cov)]])
    Sigma_mix = Sigma_mean + beta_cov

    f_lam_bar = float(np.mean([s["f_lam"] for s in samples]))
    f_kap_bar = float(np.mean([s["f_kap"] for s in samples]))
    _set_scales(blocks, base_lams, base_kaps, f_lam_bar, f_kap_bar)
    a = _assemble_precision(blocks)
    _, _, Sigma_f = _logistic_log_evidence(Z, y, blocks, beta0=beta_mean)
    edf = {}
    for b in blocks:
        if isinstance(b, _InterceptSpec):
            continue
        edf[b.name] = b.edf(a[b.sl], np.diag(Sigma_f)[b.sl])

    out = dict(
        beta=beta_mean,
        Sigma=Sigma_mix,
        evidence=float(np.mean([s["log_evidence"] for s in samples])),
        history=map_fit["history"],
        edf=edf,
        n_iter=map_fit["n_iter"],
        samples=samples,
        accept_rate=accept_rate,
        f_lam_mean=f_lam_bar,
        f_kap_mean=f_kap_bar,
        map_evidence=map_fit["evidence"],
        base_lams=base_lams,
        base_kaps=base_kaps,
        n_samples=len(samples),
        base_mode=base_mode,
        ess=ess,
        rhat=rhat,
        n_chains=len(per_chain),
        step_adapted=steps,
    )
    for k in ("total_edf", "edf_budget", "cal_scale_factor", "n_pruned", "a_floor"):
        if k in map_fit:
            out[f"map_{k}"] = map_fit[k]
    return out


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

class _SIMixin:
    def __init__(
        self,
        n_knots=10,
        max_linear_card=5,
        feature_types=None,
        interactions="auto",
        max_interactions=8,
        max_screen_pairs=40,
        interaction_gain_threshold=4.0,
        max_iter=200,
        tol=1e-5,
        feature_names=None,
        verbose=False,
        n_samples=2000,
        n_burn=1000,
        thin=1,
        n_chains=4,
        mh_step=0.35,
        adapt_step=True,
        tau_prior=1.0,
        log_f_prior_mean=0.0,
        min_f=1.0,
        max_f=50.0,
        sample_sigma=True,
        base="type2",
        # pooled-stage knobs (used when base='pooled')
        pool_strength=2.0,
        null_pool_strength=2.0,
        floor_scale=1.0,
        max_total_edf="auto",
        prune_edf=0.15,
        mackay=False,
        mackay_mix=0.35,
        linear_first=False,
        holdout_calibrate=True,
        holdout_frac=0.25,
        loo_calibrate=False,
        random_state=0,
    ):
        super().__init__(
            n_knots=n_knots,
            max_linear_card=max_linear_card,
            feature_types=feature_types,
            interactions=interactions,
            max_interactions=max_interactions,
            max_screen_pairs=max_screen_pairs,
            interaction_gain_threshold=interaction_gain_threshold,
            max_iter=max_iter,
            tol=tol,
            feature_names=feature_names,
            verbose=verbose,
        )
        self.n_samples = n_samples
        self.n_burn = n_burn
        self.thin = thin
        self.n_chains = n_chains
        self.mh_step = mh_step
        self.adapt_step = adapt_step
        self.tau_prior = tau_prior
        self.log_f_prior_mean = log_f_prior_mean
        self.min_f = min_f
        self.max_f = max_f
        self.sample_sigma = sample_sigma
        self.base = base
        self.pool_strength = pool_strength
        self.null_pool_strength = null_pool_strength
        self.floor_scale = floor_scale
        self.max_total_edf = max_total_edf
        self.prune_edf = prune_edf
        self.mackay = mackay
        self.mackay_mix = mackay_mix
        self.linear_first = linear_first
        self.holdout_calibrate = holdout_calibrate
        self.holdout_frac = holdout_frac
        self.loo_calibrate = loo_calibrate
        self.random_state = random_state

    def _pooled_kwargs(self):
        return dict(
            pool_strength=self.pool_strength,
            null_pool_strength=self.null_pool_strength,
            floor_scale=self.floor_scale,
            max_total_edf=self.max_total_edf,
            prune_edf=self.prune_edf,
            mackay=self.mackay,
            mackay_mix=self.mackay_mix,
            linear_first=self.linear_first,
            holdout_calibrate=self.holdout_calibrate,
            holdout_frac=self.holdout_frac,
            loo_calibrate=self.loo_calibrate,
            random_state=self.random_state,
        )

    def _si_kwargs(self):
        return dict(
            n_samples=self.n_samples,
            n_burn=self.n_burn,
            thin=self.thin,
            n_chains=self.n_chains,
            step=self.mh_step,
            adapt_step=self.adapt_step,
            tau_prior=self.tau_prior,
            log_f_prior_mean=self.log_f_prior_mean,
            min_f=self.min_f,
            max_f=self.max_f,
            base=self.base,
            pooled_kwargs=self._pooled_kwargs() if str(self.base).lower() in ("pooled", "pool") else None,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def _store_si_diagnostics(self, fit):
        self.samples_ = fit.get("samples", [])
        self.accept_rate_ = fit.get("accept_rate")
        self.f_lam_mean_ = fit.get("f_lam_mean")
        self.f_kap_mean_ = fit.get("f_kap_mean")
        self.f_sig_mean_ = fit.get("f_sig_mean", 1.0)
        self.map_evidence_ = fit.get("map_evidence")
        self.n_posterior_samples_ = fit.get("n_samples", len(self.samples_))
        self.ess_ = fit.get("ess", {})
        self.rhat_ = fit.get("rhat", {})
        self.n_chains_ = fit.get("n_chains")
        self.mh_step_adapted_ = fit.get("step_adapted")
        self.min_ess_ = min(self.ess_.values()) if self.ess_ else None
        _finite = [v for v in self.rhat_.values() if np.isfinite(v)]
        self.max_rhat_ = max(_finite) if _finite else None
        self.base_mode_ = fit.get("base_mode", self.base)
        self.total_edf_ = float(sum(fit.get("edf", {}).values())) if fit.get("edf") else None
        # pooled MAP diagnostics (if stage-1 was pooled)
        self.map_total_edf_ = fit.get("map_total_edf")
        self.map_edf_budget_ = fit.get("map_edf_budget")
        self.map_cal_scale_factor_ = fit.get("map_cal_scale_factor")
        self.map_n_pruned_ = fit.get("map_n_pruned")

    def _aggregate_si_diagnostics_from_estimators(self):
        """Surface worst-case sampler diagnostics from one-vs-rest sub-models.

        Each class runs an independent SI sampler. The parent reports
        averages for means / acceptance, and **worst-case** ESS / R-hat so
        ``min_ess_`` / ``max_rhat_`` remain a valid gate before quoting
        multiclass uncertainty.
        """
        ests = self.estimators_
        self.accept_rate_ = float(np.mean([
            getattr(e, "accept_rate_", 0.0) or 0.0 for e in ests
        ]))
        self.f_lam_mean_ = float(np.mean([
            getattr(e, "f_lam_mean_", 1.0) or 1.0 for e in ests
        ]))
        self.f_kap_mean_ = float(np.mean([
            getattr(e, "f_kap_mean_", 1.0) or 1.0 for e in ests
        ]))
        self.n_posterior_samples_ = int(np.mean([
            getattr(e, "n_posterior_samples_", 0) or 0 for e in ests
        ]))
        self.base_mode_ = getattr(ests[0], "base_mode_", self.base)
        self.n_chains_ = getattr(ests[0], "n_chains_", self.n_chains)
        self.mh_step_adapted_ = [
            getattr(e, "mh_step_adapted_", None) for e in ests
        ]
        self.map_evidence_ = float(sum(
            getattr(e, "map_evidence_", 0.0) or 0.0 for e in ests
        ))

        keys = set()
        for e in ests:
            keys.update(getattr(e, "ess_", {}) or {})
        ess, rhat = {}, {}
        for k in keys:
            ess_vals = [
                e.ess_[k] for e in ests
                if getattr(e, "ess_", None) and k in e.ess_
            ]
            rhat_vals = [
                e.rhat_[k] for e in ests
                if getattr(e, "rhat_", None) and k in e.rhat_
                and np.isfinite(e.rhat_[k])
            ]
            if ess_vals:
                ess[k] = float(min(ess_vals))
            if rhat_vals:
                rhat[k] = float(max(rhat_vals))
        self.ess_ = ess
        self.rhat_ = rhat
        self.min_ess_ = min(ess.values()) if ess else None
        self.max_rhat_ = max(rhat.values()) if rhat else None


def _si_constructor_kwargs(obj):
    """Copy SI (+ optional pooled) constructor fields for multiclass clones."""
    return dict(
        n_knots=obj.n_knots,
        max_linear_card=obj.max_linear_card,
        feature_types=obj.feature_types,
        interactions=obj.interactions,
        max_interactions=obj.max_interactions,
        max_screen_pairs=obj.max_screen_pairs,
        interaction_gain_threshold=obj.interaction_gain_threshold,
        max_iter=obj.max_iter,
        tol=obj.tol,
        feature_names=list(obj.feature_names) if obj.feature_names is not None else None,
        verbose=obj.verbose,
        n_samples=obj.n_samples,
        n_burn=obj.n_burn,
        thin=obj.thin,
        n_chains=obj.n_chains,
        mh_step=obj.mh_step,
        adapt_step=obj.adapt_step,
        tau_prior=obj.tau_prior,
        log_f_prior_mean=obj.log_f_prior_mean,
        min_f=obj.min_f,
        max_f=obj.max_f,
        sample_sigma=obj.sample_sigma,
        base=obj.base,
        pool_strength=obj.pool_strength,
        null_pool_strength=obj.null_pool_strength,
        floor_scale=obj.floor_scale,
        max_total_edf=obj.max_total_edf,
        prune_edf=obj.prune_edf,
        mackay=obj.mackay,
        mackay_mix=obj.mackay_mix,
        linear_first=obj.linear_first,
        holdout_calibrate=obj.holdout_calibrate,
        holdout_frac=obj.holdout_frac,
        loo_calibrate=obj.loo_calibrate,
        random_state=obj.random_state,
    )


class SRAERegressorSI(_SIMixin, SRAERegressor):
    """SRAE regression with integration over global hyperparameter scales.

    Empirical Bayes conditions on a single maximizing hyperparameter vector
    :math:`\\theta^{*}`, which is over-confident at small :math:`n`. This
    estimator keeps the *relative* per-block scales fixed at
    :math:`\\theta^{*}` and places a posterior on global multipliers

    .. math::

        \\lambda_j = f_\\lambda \\lambda_j^{*}, \\qquad
        \\kappa_j = f_\\kappa \\kappa_j^{*}, \\qquad
        \\sigma^2 = f_\\sigma \\sigma^{2*},

    with prior :math:`\\log f \\sim \\mathcal{N}(\\texttt{log_f_prior_mean},
    \\texttt{tau_prior}^2)`. A random-walk Metropolis-Hastings sampler on
    :math:`\\log f` targets :math:`p(f \\mid y) \\propto
    \\exp(\\mathrm{evidence}(f))\\,p(f)`, and predictions are the Monte Carlo
    average over draws.

    The relative per-block scales stay fixed at the empirical-Bayes estimate,
    so this is a scale mixture conditional on that estimate, not full Bayes
    over the hyperparameters.

    Read more in the :ref:`User Guide <axis_hyperprior>`.

    Parameters
    ----------
    n_knots : int, default=10
        Interior knots per spline block.

    max_linear_card : int, default=5
        Auto-dispatch only: features with at most this many distinct values
        receive a :class:`~srae.blocks.LinearBlock`. Prefer
        ``feature_types='factor'`` for nominal or non-monotone low-cardinality
        effects; ``0`` disables the auto rule.

    feature_types : None, str, sequence of str, or dict, default=None
        Per-feature block choice. See :class:`~srae.SRAERegressor`.

    interactions : {'auto', True, False}, default='auto'
        Whether to screen for pairwise interactions.

    max_interactions : int, default=8
        Maximum number of retained interaction surfaces.

    max_screen_pairs : int, default=40
        Candidate pairs scored by full evidence.

    interaction_gain_threshold : float, default=4.0
        Minimum evidence gain, in nats, to retain a pair.

    max_iter : int, default=200
        Maximum EM iterations for the stage-1 MAP fit.

    tol : float, default=1e-5
        Relative evidence tolerance for stage 1.

    feature_names : list of str, default=None
        Component labels. Never modified by ``fit``.

    verbose : bool, default=False
        Print the MAP and sampler traces.

    n_samples : int, default=2000
        Retained posterior draws **per chain**. Cost is roughly
        ``n_chains * (n_burn + n_samples * thin)`` posterior solves per fit.
        Defaults aim to clear ``ess_`` >= 100 on many smooth additive
        problems, but that is not guaranteed: the ``f_lam`` posterior is
        heavy-tailed, and hard designs can still warn. Always check
        ``min_ess_`` / ``max_rhat_`` before quoting uncertainty; far fewer
        draws than this can leave Monte Carlo error comparable to the
        posterior spread itself.

    n_burn : int, default=1000
        Burn-in iterations discarded per chain. The proposal scale adapts
        during burn-in only (see ``adapt_step``).

    thin : int, default=1
        Keep every ``thin``-th draw after burn-in. Thinning lowers ``ess_``
        and is only worth it to bound memory.

    n_chains : int, default=4
        Independent chains, seeded deterministically from ``random_state``.
        More than one is required for ``rhat_`` to exist at all.

    mh_step : float, default=0.35
        Initial random-walk standard deviation on :math:`\\log f`. With
        ``adapt_step=True`` this is only a starting point.

    adapt_step : bool, default=True
        Tune the proposal scale during burn-in toward ~0.234 acceptance by
        Robbins-Monro updates. Adaptation stops before the first retained
        draw, so the sampled chain has a fixed kernel. Disable to sample with
        exactly ``mh_step``.

    tau_prior : float, default=1.0
        Prior standard deviation on :math:`\\log f`.

    log_f_prior_mean : float, default=0.0
        Prior mean on :math:`\\log f`.

    min_f : float, default=1.0
        Lower bound on the scale factors. The default restricts sampling to
        models *at least as regularized* as the MAP. This is a truncated
        (informative) prior, not an approximation: the posterior is exact
        under it. Set near 0 for an untruncated two-sided scale posterior.

    max_f : float, default=50.0
        Upper bound on the scale factors.

    sample_sigma : bool, default=True
        Also integrate over the residual-variance multiplier
        :math:`f_\\sigma`.

    base : {'type2', 'pooled'}, default='type2'
        Stage-1 MAP used as the unit scale :math:`f = 1`. ``'pooled'`` runs
        the hierarchical stack first; see :class:`SRAERegressorSIPooled`.

    pool_strength, null_pool_strength, floor_scale, max_total_edf, prune_edf,\
 mackay, mackay_mix, linear_first, holdout_calibrate, holdout_frac,\
 loo_calibrate : optional
        Stage-1 pooled-stack settings. Ignored unless ``base='pooled'``. See
        :class:`~srae.SRAERegressorPooled` for their meaning.

    random_state : int, default=0
        Seed for the Metropolis-Hastings chain and any internal splits.

    Attributes
    ----------
    samples_ : list of dict
        One entry per retained MCMC draw. Every entry carries the scale
        factors and ``log_evidence``; only a thinned subsample of at most
        ``_MAX_STORED_DRAWS`` (128) entries also carries ``'beta'`` and
        ``'Sigma'``. ``Sigma`` is ``q x q``, so keeping one per draw would
        need several GB at the default draw count. Thinning bounds memory;
        the predictive Monte Carlo average then uses that finite subsample
        (still a valid estimator, with larger Monte Carlo error than using
        every draw). ESS and Rhat are always computed from the complete set
        of scale draws.
    accept_rate_ : float
        Metropolis acceptance rate over the *retained* draws, averaged across
        chains. The burn-in rate is excluded because adaptation contaminates
        it. Far from 0.2-0.5 with ``adapt_step=True`` indicates a difficult
        posterior rather than a tuning problem.

    ess_, rhat_ : dict of str to float
        Effective sample size (summed over chains) and split-Rhat for each
        scale factor. A ``RuntimeWarning`` is raised at fit time when
        ``ess_`` falls below 100 or ``rhat_`` exceeds 1.05.

    min_ess_, max_rhat_ : float
        Worst value across scale factors — the two numbers to report when
        quoting any uncertainty from this estimator.

    n_chains_ : int
        Chains actually run.

    mh_step_adapted_ : list of float
        Final proposal scale per chain. Compare with ``mh_step`` to see how
        far adaptation had to move.

    f_lam_mean_, f_kap_mean_, f_sig_mean_ : float
        Posterior means of the scale factors. Values near 1 mean the sampler
        found no reason to depart from the MAP.

    map_evidence_ : float
        Evidence at the stage-1 MAP.

    n_posterior_samples_ : int
        Number of retained draws.

    base_mode_ : str
        Which stage-1 MAP was used.

    Sigma_ : ndarray
        Mixture covariance :math:`\\mathbb{E}_f[\\Sigma_f] +
        \\operatorname{Var}_f[\\beta_f]`, by the law of total variance.

    See Also
    --------
    SRAERegressor : Point-estimate hyperparameters.
    SRAERegressorSIPooled : Same integration on top of the pooled MAP.

    Notes
    -----
    ``evidence_`` reports the *mean log evidence* over draws. Because
    ``min_f=1`` restricts the posterior to :math:`f \\ge 1`, this lies below
    the :math:`f = 1` peak by construction and is not comparable with
    :class:`~srae.SRAERegressor`. It is also not the marginal likelihood of
    the scale-extended model. Compare variants by held-out score only.

    Examples
    --------
    >>> import numpy as np
    >>> from srae import SRAERegressorSI
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 3))
    >>> y = np.sin(1.5 * X[:, 0]) + 0.2 * rng.normal(size=120)
    >>> m = SRAERegressorSI(interactions=False, n_samples=8, n_burn=4).fit(X, y)
    >>> bool(0.0 <= m.accept_rate_ <= 1.0)
    True
    """

    def _fit_engine(self, Z, y):
        self._ymean = float(np.mean(y))
        self._n_train = int(len(y))
        kw = self._si_kwargs()
        kw["sample_sigma"] = self.sample_sigma
        return fit_gaussian_si(Z, y - self._ymean, self.specs_, **kw)

    def _finalize(self, y, fit):
        super()._finalize(y, fit)
        self._store_si_diagnostics(fit)

    def predict(self, X, return_std=False):
        """Monte Carlo average over hyperparameter posterior samples.

        Predictive variance uses the law of total variance over draws of
        :math:`f`, plus the intercept sampling term
        :math:`\\sigma^2(f) / n_{\\mathrm{train}}` on each draw (same as
        :meth:`~srae.SRAERegressor.predict`).
        """
        Z = self._design(self._coerce(X))
        samples = _array_draws(getattr(self, "samples_", None) or [])
        if not samples:
            return super().predict(X, return_std=return_std)

        means = []
        vars_ = []
        for s in samples:
            m = self._ymean + Z @ s["beta"]
            vf = np.einsum("ij,jk,ik->i", Z, s["Sigma"], Z)
            means.append(m)
            # Per-draw noise + intercept sampling variance of ybar.
            vars_.append(
                np.clip(vf, 0, None)
                + s["sigma2"]
                + self._intercept_sampling_variance(s["sigma2"])
            )
        means = np.stack(means, axis=0)
        vars_ = np.stack(vars_, axis=0)
        mean = means.mean(axis=0)
        if not return_std:
            return mean
        var = vars_.mean(axis=0) + means.var(axis=0)
        return mean, np.sqrt(np.clip(var, 0, None))


class SRAEClassifierSI(_SIMixin, SRAEClassifier):
    """SRAE classification with integration over hyperparameter scales.

    As :class:`SRAERegressorSI`, but for the Bernoulli-logit likelihood and
    without a residual-variance multiplier: the sampler moves over
    :math:`(f_\\lambda, f_\\kappa)` only. For a binary fit,
    :meth:`predict_proba` averages the moderated probabilities
    :math:`\\operatorname{sigmoid}(\\mu / \\sqrt{1 + \\pi\\nu/8})` across
    posterior draws. For multiclass, it pairs one posterior logit draw from
    each independently fitted one-vs-rest head, applies a softmax, and averages
    the resulting probability vectors.

    Read more in the :ref:`User Guide <axis_hyperprior>`.

    Parameters
    ----------
    n_knots : int, default=10
        Interior knots per spline block.

    max_linear_card : int, default=5
        Auto-dispatch only: features with at most this many distinct values
        receive a :class:`~srae.blocks.LinearBlock`. Prefer
        ``feature_types='factor'`` for nominal or non-monotone low-cardinality
        effects; ``0`` disables the auto rule.

    feature_types : None, str, sequence of str, or dict, default=None
        Per-feature block choice. See :class:`~srae.SRAERegressor`.

    interactions : {'auto', True, False}, default='auto'
        Whether to screen for pairwise interactions.

    max_interactions : int, default=8
        Maximum number of retained interaction surfaces.

    max_screen_pairs : int, default=40
        Candidate pairs scored by full evidence.

    interaction_gain_threshold : float, default=4.0
        Minimum evidence gain, in nats, to retain a pair.

    max_iter : int, default=200
        Maximum iterations for the stage-1 MAP; capped at 100 internally.

    tol : float, default=1e-5
        Relative evidence tolerance for stage 1.

    feature_names : list of str, default=None
        Component labels. Never modified by ``fit``.

    verbose : bool, default=False
        Print the MAP and sampler traces.

    n_samples : int, default=2000
        Retained posterior draws **per chain**. Cost is roughly
        ``n_chains * (n_burn + n_samples * thin)`` posterior solves per
        binary fit; multiclass multiplies that by the number of classes.
        Defaults aim to clear ``ess_`` >= 100 on many smooth additive
        problems, but that is not guaranteed — always check ``min_ess_``
        and ``max_rhat_``.

    n_burn : int, default=1000
        Burn-in iterations discarded per chain. The proposal scale adapts
        during burn-in only (see ``adapt_step``).

    thin : int, default=1
        Keep every ``thin``-th draw after burn-in. Thinning lowers ``ess_``
        and is only worth it to bound memory.

    n_chains : int, default=4
        Independent chains, seeded deterministically from ``random_state``.
        More than one is required for ``rhat_`` to exist at all.

    mh_step : float, default=0.35
        Initial random-walk standard deviation on :math:`\\log f`. With
        ``adapt_step=True`` this is only a starting point.

    adapt_step : bool, default=True
        Tune the proposal scale during burn-in toward ~0.234 acceptance by
        Robbins-Monro updates. Adaptation stops before the first retained
        draw, so the sampled chain has a fixed kernel. Disable to sample with
        exactly ``mh_step``.

    tau_prior : float, default=1.0
        Prior standard deviation on :math:`\\log f`.

    log_f_prior_mean : float, default=0.0
        Prior mean on :math:`\\log f`.

    min_f : float, default=1.0
        Lower bound on the scale factors. The default truncates the prior to
        :math:`f \\ge 1`; the posterior is exact under that truncated prior.

    max_f : float, default=50.0
        Upper bound on the scale factors.

    sample_sigma : bool, default=True
        Retained for signature compatibility with the regressor; unused, since
        the logistic likelihood has no residual-variance parameter.

    base : {'type2', 'pooled'}, default='type2'
        Stage-1 MAP used as the unit scale.

    pool_strength, null_pool_strength, floor_scale, max_total_edf, prune_edf,\
 mackay, mackay_mix, linear_first, holdout_calibrate, holdout_frac,\
 loo_calibrate : optional
        Stage-1 pooled-stack settings. Ignored unless ``base='pooled'``.

    random_state : int, default=0
        Seed for the Metropolis-Hastings chains and any internal splits.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    estimators_ : list or None
        One-vs-rest sub-models for multiclass; ``None`` for binary.

    samples_ : list of dict
        One entry per retained MCMC draw. Every entry carries the scale
        factors and ``log_evidence``; only a thinned subsample of at most
        ``_MAX_STORED_DRAWS`` (128) entries also carries ``'beta'`` and
        ``'Sigma'``. ``Sigma`` is ``q x q``, so keeping one per draw would
        need several GB at the default draw count. Thinning bounds memory;
        the predictive Monte Carlo average then uses that finite subsample
        (still a valid estimator, with larger Monte Carlo error than using
        every draw). ESS and Rhat are always computed from the complete set
        of scale draws. Binary models only; for multiclass, coefficient draws
        stay on each object in ``estimators_``.

    accept_rate_ : float
        Metropolis acceptance rate; averaged over sub-models for multiclass.

    ess_, rhat_ : dict of str to float
        Effective sample size and split-Rhat per scale factor. Multiclass
        reports the **worst** value across one-vs-rest sub-models (min ESS,
        max R-hat) so a single gate remains valid.

    min_ess_, max_rhat_ : float
        Worst-case summaries over ``ess_`` / ``rhat_``. A
        ``RuntimeWarning`` is raised per binary fit when thresholds fail;
        for multiclass check these on the parent after fit as well.

    n_chains_ : int
        Number of chains used in each binary sub-fit.

    f_lam_mean_, f_kap_mean_ : float
        Posterior means of the scale factors (averaged over classes for
        multiclass).

    map_evidence_ : float
        Evidence at the stage-1 MAP (summed over classes for multiclass).

    See Also
    --------
    SRAEClassifier : Point-estimate hyperparameters.
    SRAEClassifierSIPooled : Same integration on top of the pooled MAP.

    Notes
    -----
    For a binary fit, ``evidence_`` is a mean log evidence over a
    truncated-prior posterior. For multiclass it is the sum of those per-head
    means. Neither is comparable with :class:`~srae.SRAEClassifier`; compare
    by held-out score.

    Cost scales with the number of classes: a multiclass fit runs a full
    multi-chain sampler (default 4 chains) once per class.
    """

    def _fit_engine(self, Z, y):
        kw = self._si_kwargs()
        kw["max_iter"] = min(int(kw.get("max_iter", 100)), 100)
        return fit_logistic_si(Z, y, self.specs_, **kw)

    def _finalize(self, y, fit):
        super()._finalize(y, fit)
        self._store_si_diagnostics(fit)

    def _si_sample_logits(self, X):
        """Per-sample linear predictors, shape ``(n_draws, n_samples)``.

        Falls back to the stage-1 MAP predictor when no draws were retained, so
        the caller always gets at least one row.
        """
        Z = self._design(self._coerce(X))
        samples = _array_draws(getattr(self, "samples_", None) or [])
        if not samples:
            return (Z @ self.beta_)[None, :]
        return np.stack([Z @ s["beta"] for s in samples], axis=0)

    def _multiclass_proba(self, X):
        """Softmax coupling that keeps the scale integration.

        The one-vs-rest heads are fitted independently, so their posteriors are
        independent and drawing one sample from each is a valid Monte Carlo draw
        from the product posterior.  Pairing draws by index, softmaxing each
        paired K-vector, and averaging the resulting probability vectors
        therefore integrates over the global scales rather than discarding them.

        Collapsing to the posterior mean logit first would not be the same
        thing: softmax is non-linear, so averaging logits and then softmaxing
        understates predictive uncertainty relative to averaging probabilities.
        """
        if getattr(self, "multiclass_link", "softmax") == "normalized_ovr":
            P = np.column_stack([e.predict_proba(X)[:, 1] for e in self.estimators_])
            return P / P.sum(axis=1, keepdims=True)

        per_head = [e._si_sample_logits(X) for e in self.estimators_]
        n_draws = min(head.shape[0] for head in per_head)
        total = None
        for d in range(n_draws):
            eta = np.column_stack([head[d] for head in per_head])
            eta = eta - eta.max(axis=1, keepdims=True)
            P = np.exp(eta)
            P /= P.sum(axis=1, keepdims=True)
            total = P if total is None else total + P
        return total / float(n_draws)

    def predict_proba(self, X):
        if self._is_multiclass:
            return self._multiclass_proba(X)

        Z = self._design(self._coerce(X))
        samples = _array_draws(getattr(self, "samples_", None) or [])
        if not samples:
            return super().predict_proba(X)

        ps = []
        for s in samples:
            mu = Z @ s["beta"]
            var = np.clip(np.einsum("ij,jk,ik->i", Z, s["Sigma"], Z), 0, None)
            p1 = _sigmoid(mu / np.sqrt(1.0 + np.pi * var / 8.0))
            ps.append(p1)
        p1 = np.mean(np.stack(ps, axis=0), axis=0)
        return np.column_stack([1.0 - p1, p1])

    def fit(self, X, y):
        self._set_sklearn_fit_attrs(X)
        Xa = self._coerce(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least 2 classes")
        if len(self.classes_) == 2:
            self.estimators_ = None
            y01 = (y == self.classes_[1]).astype(float)
            _BaseSRAE.fit(self, X, y01)
            return self

        self.estimators_ = []
        cls = type(self)
        for c in self.classes_:
            est = cls(**_si_constructor_kwargs(self))
            est.feature_names = list(self.feature_names_)
            est.fit(Xa, (y == c).astype(float))
            self.estimators_.append(est)
        self.interactions_ = [
            dict(cls=c, **info)
            for c, est in zip(self.classes_, self.estimators_)
            for info in est.interactions_
        ]
        self.evidence_ = float(sum(e.evidence_ for e in self.estimators_))
        self._aggregate_si_diagnostics_from_estimators()
        return self


class SRAERegressorSIPooled(SRAERegressorSI):
    """SRAE regression: pooled MAP followed by scale integration.

    Both axes at once. Stage 1 runs the hierarchical anti-overfitting stack of
    :class:`~srae.SRAERegressorPooled` -- pooling, precision floors, soft
    pruning, an edf budget and holdout calibration. Stage 2 then integrates
    global scale factors :math:`(f_\\lambda, f_\\kappa, f_\\sigma)` relative to
    *that* base, as in :class:`SRAERegressorSI`.

    Behaviourally equivalent to ``SRAERegressorSI(base='pooled', ...)``; the
    class exists so that :func:`sklearn.base.clone` and ``get_params`` see an
    explicit signature rather than ``**kwargs``, which scikit-learn would
    silently drop.

    Read more in the :ref:`User Guide <variants>`.

    Parameters
    ----------
    base : {'pooled', 'type2'}, default='pooled'
        Stage-1 MAP. Defaults to ``'pooled'``; this is the only difference
        from :class:`SRAERegressorSI`.

    **kwargs : see below
        All other parameters are as documented on
        :class:`SRAERegressorSI` (sampler settings) and
        :class:`~srae.SRAERegressorPooled` (stage-1 stack settings). Unlike on
        :class:`SRAERegressorSI`, the pooled settings are *active* here.

    Attributes
    ----------
    map_total_edf_ : float
        Total edf reached by the stage-1 pooled MAP.

    map_edf_budget_ : float or None
        The stage-1 edf cap.

    map_cal_scale_factor_ : float
        Stage-1 holdout calibration multiplier.

    map_n_pruned_ : int
        Blocks pinned during stage 1.

    See Also
    --------
    SRAERegressorSI : Scale integration on the Type-II MAP.
    SRAERegressorPooled : The pooled MAP without integration.

    Notes
    -----
    Stacking both axes compounds their shrinkage. Since the pooled stack alone
    can cost substantial fit quality at small :math:`n`, and ``min_f=1``
    only shrinks further, verify against the plain estimator rather than
    assuming this is the safest choice.
    """

    def __init__(
        self,
        n_knots=10,
        max_linear_card=5,
        feature_types=None,
        interactions="auto",
        max_interactions=8,
        max_screen_pairs=40,
        interaction_gain_threshold=4.0,
        max_iter=200,
        tol=1e-5,
        feature_names=None,
        verbose=False,
        n_samples=2000,
        n_burn=1000,
        thin=1,
        n_chains=4,
        mh_step=0.35,
        adapt_step=True,
        tau_prior=1.0,
        log_f_prior_mean=0.0,
        min_f=1.0,
        max_f=50.0,
        sample_sigma=True,
        base="pooled",
        pool_strength=2.0,
        null_pool_strength=2.0,
        floor_scale=1.0,
        max_total_edf="auto",
        prune_edf=0.15,
        mackay=False,
        mackay_mix=0.35,
        linear_first=False,
        holdout_calibrate=True,
        holdout_frac=0.25,
        loo_calibrate=False,
        random_state=0,
    ):
        super().__init__(
            n_knots=n_knots,
            max_linear_card=max_linear_card,
            feature_types=feature_types,
            interactions=interactions,
            max_interactions=max_interactions,
            max_screen_pairs=max_screen_pairs,
            interaction_gain_threshold=interaction_gain_threshold,
            max_iter=max_iter,
            tol=tol,
            feature_names=feature_names,
            verbose=verbose,
            n_samples=n_samples,
            n_burn=n_burn,
            thin=thin,
            n_chains=n_chains,
            mh_step=mh_step,
            adapt_step=adapt_step,
            tau_prior=tau_prior,
            log_f_prior_mean=log_f_prior_mean,
            min_f=min_f,
            max_f=max_f,
            sample_sigma=sample_sigma,
            base=base,
            pool_strength=pool_strength,
            null_pool_strength=null_pool_strength,
            floor_scale=floor_scale,
            max_total_edf=max_total_edf,
            prune_edf=prune_edf,
            mackay=mackay,
            mackay_mix=mackay_mix,
            linear_first=linear_first,
            holdout_calibrate=holdout_calibrate,
            holdout_frac=holdout_frac,
            loo_calibrate=loo_calibrate,
            random_state=random_state,
        )


class SRAEClassifierSIPooled(SRAEClassifierSI):
    """SRAE classification: pooled MAP followed by scale integration.

    Stage 1 runs the hierarchical stack of
    :class:`~srae.SRAEClassifierPooled` -- pooling, precision floors, soft
    pruning, an edf budget and holdout calibration. Stage 2 samples global
    scale factors :math:`(f_\\lambda, f_\\kappa)` with :math:`f \\ge`
    ``min_f`` relative to that base. Binary predictions average moderated
    probabilities; multiclass predictions average paired-draw softmax
    probabilities.

    Behaviourally equivalent to ``SRAEClassifierSI(base='pooled', ...)``; the
    explicit signature exists so scikit-learn's ``get_params`` / ``clone``
    work correctly.

    Read more in the :ref:`User Guide <variants>`.

    Parameters
    ----------
    base : {'pooled', 'type2'}, default='pooled'
        Stage-1 MAP. Defaults to ``'pooled'``.

    **kwargs : see below
        All other parameters are as documented on
        :class:`SRAEClassifierSI` (sampler settings) and
        :class:`~srae.SRAEClassifierPooled` (stage-1 stack settings), the
        latter being active here.

    Attributes
    ----------
    map_total_edf_ : float
        Total edf reached by the stage-1 pooled MAP. Binary models only; for
        multiclass inspect each object in ``estimators_``.

    map_edf_budget_ : float or None
        The stage-1 edf cap. Binary models only; for multiclass inspect each
        object in ``estimators_``.

    map_cal_scale_factor_ : float
        Stage-1 holdout calibration multiplier. Binary models only; for
        multiclass inspect each object in ``estimators_``.

    map_n_pruned_ : int
        Blocks pinned during stage 1. Binary models only; for multiclass
        inspect each object in ``estimators_``.

    See Also
    --------
    SRAEClassifierSI : Scale integration on the Type-II MAP.
    SRAEClassifierPooled : The pooled MAP without integration.

    Notes
    -----
    This is the most heavily regularized of the eight estimators and the most
    expensive: the classification edf budget is the tightest, and a multiclass
    fit runs a pooled stage-1 fit plus a full multi-chain sampler per class.
    Verify against the plain estimator on held-out data.
    """

    def __init__(
        self,
        n_knots=10,
        max_linear_card=5,
        feature_types=None,
        interactions="auto",
        max_interactions=8,
        max_screen_pairs=40,
        interaction_gain_threshold=4.0,
        max_iter=200,
        tol=1e-5,
        feature_names=None,
        verbose=False,
        n_samples=2000,
        n_burn=1000,
        thin=1,
        n_chains=4,
        mh_step=0.35,
        adapt_step=True,
        tau_prior=1.0,
        log_f_prior_mean=0.0,
        min_f=1.0,
        max_f=50.0,
        sample_sigma=True,
        base="pooled",
        pool_strength=2.0,
        null_pool_strength=2.0,
        floor_scale=1.0,
        max_total_edf="auto",
        prune_edf=0.15,
        mackay=False,
        mackay_mix=0.35,
        linear_first=False,
        holdout_calibrate=True,
        holdout_frac=0.25,
        loo_calibrate=False,
        random_state=0,
    ):
        super().__init__(
            n_knots=n_knots,
            max_linear_card=max_linear_card,
            feature_types=feature_types,
            interactions=interactions,
            max_interactions=max_interactions,
            max_screen_pairs=max_screen_pairs,
            interaction_gain_threshold=interaction_gain_threshold,
            max_iter=max_iter,
            tol=tol,
            feature_names=feature_names,
            verbose=verbose,
            n_samples=n_samples,
            n_burn=n_burn,
            thin=thin,
            n_chains=n_chains,
            mh_step=mh_step,
            adapt_step=adapt_step,
            tau_prior=tau_prior,
            log_f_prior_mean=log_f_prior_mean,
            min_f=min_f,
            max_f=max_f,
            sample_sigma=sample_sigma,
            base=base,
            pool_strength=pool_strength,
            null_pool_strength=null_pool_strength,
            floor_scale=floor_scale,
            max_total_edf=max_total_edf,
            prune_edf=prune_edf,
            mackay=mackay,
            mackay_mix=mackay_mix,
            linear_first=linear_first,
            holdout_calibrate=holdout_calibrate,
            holdout_frac=holdout_frac,
            loo_calibrate=loo_calibrate,
            random_state=random_state,
        )
