"""Pooled / hierarchical-regularization SRAE (benchmarking variant).

Separate from the default Type-II MLE path.  Original ``SRAERegressor`` /
``SRAEClassifier`` are unchanged.

Anti-overfitting stack (defaults tuned for small-n / high-p):

1. **Conservative partial pooling** of λ / κ across blocks (only ever more
   regularized than the per-block Type-II update), with optional stronger
   pooling on null-space (linear) ARD.
2. **Precision floors** ``a_i >= floor_scale / n``.
3. **Soft block pruning** — blocks with edf below ``prune_edf`` are pinned
   to near-zero and the posterior is recomputed.
4. **Auto (or explicit) total-edf budget** via a global precision rescale.
5. **Holdout global calibration** — pick a global precision multiplier by
   maximizing held-out log predictive density on an internal split of the
   training data (stratified for classification; random fallback for
   continuous regression). Steps 1-4 have already used every row, so this
   calibrates one scalar against hyperparameters fitted in-sample; it is not
   held-out validation. The calibrated main-effect fit also supplies the
   residuals used for interaction screening, and calibration runs again after
   a selected interaction refit. See ``holdout_calibrate``.
6. Optional **MacKay γ-updates**, **linear-first two-stage**, and **approx
   LOO calibration** (off by default; useful for ablation).

Drop-in::

    from srae import SRAEClassifier, SRAEClassifierPooled
    m = SRAEClassifierPooled(feature_names=names, interactions=False, n_knots=8)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .inference import (
    _assemble_precision,
    _map_logistic,
    _sigmoid,
    _A_MIN,
    _A_MAX,
)
from .model import SRAEClassifier, SRAERegressor, _InterceptSpec

__all__ = [
    "SRAERegressorPooled",
    "SRAEClassifierPooled",
    "fit_gaussian_eb_pooled",
    "fit_logistic_eb_pooled",
]


# ---------------------------------------------------------------------------
# Hyperparameter helpers
# ---------------------------------------------------------------------------

def _pooled_map(count, denom, pool_strength, prior_mean):
    s = float(max(pool_strength, 0.0))
    d = max(float(denom), 1e-300)
    if s <= 0.0:
        return float(np.clip(count / d, _A_MIN, _A_MAX))
    m = max(float(prior_mean), 1e-300)
    return float(np.clip((count + 2.0 * s) / (d + 2.0 * s / m), _A_MIN, _A_MAX))


def _auto_edf_budget(n, task="logistic"):
    """Capacity budget that scales with sample size.

    Classification (high-p ICP-like): ~0.5 √n  (aggressive but keeps val AUC)
    Regression: slightly looser ~0.85 √n
    """
    n = max(int(n), 2)
    root = np.sqrt(n)
    if task == "logistic":
        return float(max(4.0, min(0.50 * root, n / max(3.5 * np.log(n), 1.0))))
    return float(max(6.0, min(0.85 * root, n / max(2.0 * np.log(n), 1.0))))


def _resolve_edf_budget(max_total_edf, n, task):
    if max_total_edf is None:
        return None
    if isinstance(max_total_edf, str):
        if max_total_edf.lower() in ("auto", "default"):
            return _auto_edf_budget(n, task=task)
        raise ValueError(f"unknown max_total_edf={max_total_edf!r}")
    return float(max_total_edf)


def _em_update_pooled(
    blocks, beta, dSig, a,
    pool_strength,
    null_pool_strength,
    a_floor,
    mackay=False,
    mackay_mix=0.35,
):
    """Type-II / optional damped-MacKay update + conservative pooling + floors."""
    lam_raw, kap_raw = [], []
    stats = []

    for b in blocks:
        if isinstance(b, _InterceptSpec):
            continue
        sl = b.sl
        bb, ss, aa = beta[sl], dSig[sl], a[sl]
        pen, null = b.pen, ~b.pen
        r, m_null = int(pen.sum()), int(null.sum())
        lam_mle = kap_mle = None

        if r:
            e2 = bb[pen] ** 2 + ss[pen]
            S_em = float(np.sum(b.s[pen] * e2))
            lam_em = r / max(S_em, 1e-300)
            if mackay:
                gamma = np.clip(1.0 - aa[pen] * ss[pen], 0.0, 1.0)
                S_mac = float(np.sum(b.s[pen] * bb[pen] ** 2))
                lam_mac = float(np.sum(gamma)) / max(S_mac, 1e-300)
                # Damped blend; never below EM (conservative).
                mix = float(np.clip(mackay_mix, 0.0, 1.0))
                lam_mle = (1.0 - mix) * lam_em + mix * max(lam_em, lam_mac)
            else:
                lam_mle = lam_em
            lam_mle = float(np.clip(lam_mle, _A_MIN, _A_MAX))
            lam_raw.append(lam_mle)

        if m_null:
            e2 = bb[null] ** 2 + ss[null]
            S_em = float(np.sum(e2))
            kap_em = m_null / max(S_em, 1e-300)
            if mackay:
                gamma = np.clip(1.0 - aa[null] * ss[null], 0.0, 1.0)
                S_mac = float(np.sum(bb[null] ** 2))
                kap_mac = float(np.sum(gamma)) / max(S_mac, 1e-300)
                mix = float(np.clip(mackay_mix, 0.0, 1.0))
                kap_mle = (1.0 - mix) * kap_em + mix * max(kap_em, kap_mac)
            else:
                kap_mle = kap_em
            kap_mle = float(np.clip(kap_mle, _A_MIN, _A_MAX))
            kap_raw.append(kap_mle)

        stats.append((b, r, m_null, lam_mle, kap_mle))

    lam_bar = float(np.quantile(lam_raw, 0.75)) if lam_raw else 1.0
    kap_bar = float(np.quantile(kap_raw, 0.75)) if kap_raw else 1.0

    for b, r, m_null, lam_mle, kap_mle in stats:
        if r and lam_mle is not None:
            e2 = beta[b.sl][b.pen] ** 2 + dSig[b.sl][b.pen]
            S = float(np.sum(b.s[b.pen] * e2))
            lam_pool = _pooled_map(r, S, pool_strength, lam_bar)
            b.lam = max(lam_mle, lam_pool)
        if m_null and kap_mle is not None:
            s_null = (null_pool_strength if null_pool_strength is not None
                      else pool_strength)
            e2 = beta[b.sl][~b.pen] ** 2 + dSig[b.sl][~b.pen]
            S = float(np.sum(e2))
            kap_pool = _pooled_map(m_null, S, s_null, kap_bar)
            b.kap = max(kap_mle, kap_pool)

    if a_floor > 0:
        for b, r, m_null, lam_mle, kap_mle in stats:
            if r:
                s_pos = b.s[b.pen]
                s_min = float(np.min(s_pos[s_pos > 0])) if np.any(s_pos > 0) else 1.0
                b.lam = max(b.lam, a_floor / max(s_min, 1e-12))
                b.lam = float(np.clip(b.lam, _A_MIN, _A_MAX))
            if m_null:
                b.kap = max(b.kap, a_floor)
                b.kap = float(np.clip(b.kap, _A_MIN, _A_MAX))

    return lam_bar, kap_bar


def _total_edf(blocks, a, dSig, scale=1.0):
    total, edf = 0.0, {}
    for b in blocks:
        if isinstance(b, _InterceptSpec):
            continue
        e = b.edf(a[b.sl], dSig[b.sl], scale=scale)
        edf[b.name] = e
        total += e
    return total, edf


def _rescale_precisions(blocks, factor):
    f = float(factor)
    for b in blocks:
        if isinstance(b, _InterceptSpec):
            continue
        if b.pen.any():
            b.lam = float(np.clip(b.lam * f, _A_MIN, _A_MAX))
        if (~b.pen).any():
            b.kap = float(np.clip(b.kap * f, _A_MIN, _A_MAX))


def _save_hypers(blocks):
    return [(b.lam, b.kap) for b in blocks]


def _restore_hypers(blocks, saved):
    for b, (lam, kap) in zip(blocks, saved):
        b.lam, b.kap = lam, kap


def _apply_factor_from_saved(blocks, saved, fac):
    _restore_hypers(blocks, saved)
    _rescale_precisions(blocks, fac)


def _freeze_wiggly(blocks, lam_freeze=1e8):
    for b in blocks:
        if isinstance(b, _InterceptSpec):
            continue
        if b.pen.any():
            b.lam = float(lam_freeze)


def _prune_small_blocks(blocks, edf, prune_edf, lam_pin=1e8, kap_pin=1e8):
    """Pin near-zero blocks so they cannot re-absorb capacity later."""
    n_pruned = 0
    if prune_edf is None or prune_edf <= 0:
        return 0
    for b in blocks:
        if isinstance(b, _InterceptSpec):
            continue
        e = edf.get(b.name, 0.0)
        if e < prune_edf:
            if b.pen.any():
                b.lam = max(b.lam, lam_pin)
            if (~b.pen).any():
                b.kap = max(b.kap, kap_pin)
            n_pruned += 1
    return n_pruned


# ---------------------------------------------------------------------------
# EDF budget
# ---------------------------------------------------------------------------

def _enforce_edf_budget_gaussian(Z, y, blocks, sigma2, max_total_edf, M=None):
    if M is None:
        M = Z.T @ Z
    Zty = Z.T @ y
    Ieye = np.eye(Z.shape[1])
    saved = _save_hypers(blocks)

    def posterior(sig2):
        a = _assemble_precision(blocks)
        H = M / sig2 + np.diag(a)
        c, low = cho_factor(H, lower=True)
        beta = cho_solve((c, low), Zty / sig2)
        Sigma = cho_solve((c, low), Ieye)
        total, edf = _total_edf(blocks, a, np.diag(Sigma))
        return beta, Sigma, total, edf

    beta, Sigma, total, edf = posterior(sigma2)
    if total <= max_total_edf + 1e-6:
        return beta, Sigma, total, edf, 1.0

    hi = 1.0
    for _ in range(60):
        hi *= 2.0
        _apply_factor_from_saved(blocks, saved, hi)
        beta, Sigma, total, edf = posterior(sigma2)
        if total <= max_total_edf:
            break

    flo, fhi, best = 1.0, hi, hi
    for _ in range(40):
        mid = 0.5 * (flo + fhi)
        _apply_factor_from_saved(blocks, saved, mid)
        beta, Sigma, total, edf = posterior(sigma2)
        if total > max_total_edf:
            flo = mid
        else:
            fhi = mid
            best = mid
    _apply_factor_from_saved(blocks, saved, best)
    beta, Sigma, total, edf = posterior(sigma2)
    return beta, Sigma, total, edf, best


def _enforce_edf_budget_logistic(Z, y, blocks, max_total_edf, beta0):
    saved = _save_hypers(blocks)

    def posterior(b0):
        a = _assemble_precision(blocks)
        beta, Sigma, ll, logdetH = _map_logistic(Z, y, a, b0)
        total, edf = _total_edf(blocks, a, np.diag(Sigma))
        return beta, Sigma, total, edf, ll, logdetH

    beta, Sigma, total, edf, ll, logdetH = posterior(beta0)
    if total <= max_total_edf + 1e-6:
        return beta, Sigma, total, edf, 1.0, ll, logdetH

    hi = 1.0
    for _ in range(60):
        hi *= 2.0
        _apply_factor_from_saved(blocks, saved, hi)
        beta, Sigma, total, edf, ll, logdetH = posterior(beta)
        if total <= max_total_edf:
            break

    flo, fhi, best = 1.0, hi, hi
    for _ in range(40):
        mid = 0.5 * (flo + fhi)
        _apply_factor_from_saved(blocks, saved, mid)
        beta, Sigma, total, edf, ll, logdetH = posterior(beta)
        if total > max_total_edf:
            flo = mid
        else:
            fhi = mid
            best = mid
    _apply_factor_from_saved(blocks, saved, best)
    beta, Sigma, total, edf, ll, logdetH = posterior(beta)
    return beta, Sigma, total, edf, best, ll, logdetH


# ---------------------------------------------------------------------------
# Holdout / LOO global calibration
# ---------------------------------------------------------------------------

def _stratified_holdout_idx(y, frac=0.25, seed=0):
    y = np.asarray(y).ravel()
    n = len(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    val_idx = []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(frac * len(idx))))
        # leave at least 1 train sample per class when possible
        if len(idx) - k < 1:
            k = max(0, len(idx) - 1)
        val_idx.append(idx[:k])
    val = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
    if val.size == 0 or val.size >= n:
        # fall back to random holdout
        perm = rng.permutation(n)
        k = max(1, int(round(frac * n)))
        val = perm[:k]
    train = np.setdiff1d(np.arange(n), val, assume_unique=False)
    return train, val


def _gaussian_holdout_score(Z, y, blocks, sigma2, tr, va, M_full=None):
    """Held-out log predictive density, including parameter uncertainty.

    The variance is ``z' Sigma z + sigma2``. This includes coefficient
    uncertainty and observation noise, but omits the public regressor's
    additional ``sigma2 / n`` sampling variance for the fitted response mean.
    Scoring under ``sigma2`` alone would be narrower still.
    """
    Ztr, ytr = Z[tr], y[tr]
    M = Ztr.T @ Ztr
    Zty = Ztr.T @ ytr
    a = _assemble_precision(blocks)
    H = M / sigma2 + np.diag(a)
    c, low = cho_factor(H, lower=True)
    beta = cho_solve((c, low), Zty / sigma2)
    Sigma = cho_solve((c, low), np.eye(Z.shape[1]))
    Zva = Z[va]
    resid = y[va] - Zva @ beta
    var = np.clip(np.einsum("ij,jk,ik->i", Zva, Sigma, Zva), 0, None) + sigma2
    return float(np.sum(-0.5 * np.log(2 * np.pi * var) - 0.5 * resid ** 2 / var))


def _logistic_holdout_score(Z, y, blocks, beta0, tr, va):
    """Held-out Bernoulli log-likelihood of the *moderated* probabilities.

    A binary :meth:`~srae.SRAEClassifier.predict_proba` call integrates the
    posterior of the linear predictor against the link via the probit
    approximation. Scoring the unmoderated MAP ``eta`` would calibrate a
    different binary quantity. In a multiclass estimator this function is
    applied independently inside each one-vs-rest sub-model; the parent
    subsequently couples logits through its multiclass link.
    """
    a = _assemble_precision(blocks)
    beta, Sigma, ll, _ = _map_logistic(Z[tr], y[tr], a, beta0)
    Zva = Z[va]
    mu = Zva @ beta
    nu = np.clip(np.einsum("ij,jk,ik->i", Zva, Sigma, Zva), 0, None)
    eta = mu / np.sqrt(1.0 + np.pi * nu / 8.0)
    return float(np.sum(y[va] * eta - np.logaddexp(0.0, eta))), beta


def _holdout_calibrate_gaussian(Z, y, blocks, sigma2, frac=0.25, seed=0, n_grid=20):
    tr, va = _stratified_holdout_idx(y, frac=frac, seed=seed)
    if len(va) < 2 or len(tr) < 5:
        return 1.0, None
    saved = _save_hypers(blocks)
    grid = np.unique(np.r_[1.0, np.geomspace(1.0, 100.0, n_grid)])
    best_f, best_s = 1.0, -np.inf
    for f in grid:
        _apply_factor_from_saved(blocks, saved, f)
        s = _gaussian_holdout_score(Z, y, blocks, sigma2, tr, va)
        if s > best_s:
            best_f, best_s = f, s
    # local refine
    lo, hi = (best_f / 2 if best_f > 1 else 1.0), best_f * 2
    for _ in range(15):
        f1 = lo + (hi - lo) / 3
        f2 = hi - (hi - lo) / 3
        _apply_factor_from_saved(blocks, saved, f1)
        s1 = _gaussian_holdout_score(Z, y, blocks, sigma2, tr, va)
        _apply_factor_from_saved(blocks, saved, f2)
        s2 = _gaussian_holdout_score(Z, y, blocks, sigma2, tr, va)
        if s1 < s2:
            lo, best_f, best_s = f1, f2, s2
        else:
            hi, best_f, best_s = f2, f1, s1
    _apply_factor_from_saved(blocks, saved, best_f)
    return float(best_f), float(best_s)


def _holdout_calibrate_logistic(Z, y, blocks, beta0, frac=0.25, seed=0, n_grid=20):
    tr, va = _stratified_holdout_idx(y, frac=frac, seed=seed)
    if len(va) < 2 or len(tr) < 5:
        return beta0, 1.0, None
    saved = _save_hypers(blocks)
    grid = np.unique(np.r_[1.0, np.geomspace(1.0, 100.0, n_grid)])
    best_f, best_s, best_b = 1.0, -np.inf, beta0
    for f in grid:
        _apply_factor_from_saved(blocks, saved, f)
        s, b = _logistic_holdout_score(Z, y, blocks, best_b, tr, va)
        if s > best_s:
            best_f, best_s, best_b = f, s, b
    lo, hi = (best_f / 2 if best_f > 1 else 1.0), best_f * 2
    for _ in range(15):
        f1 = lo + (hi - lo) / 3
        f2 = hi - (hi - lo) / 3
        _apply_factor_from_saved(blocks, saved, f1)
        s1, b1 = _logistic_holdout_score(Z, y, blocks, best_b, tr, va)
        _apply_factor_from_saved(blocks, saved, f2)
        s2, b2 = _logistic_holdout_score(Z, y, blocks, best_b, tr, va)
        if s1 < s2:
            lo, best_f, best_s, best_b = f1, f2, s2, b2
        else:
            hi, best_f, best_s, best_b = f2, f1, s1, b1
    _apply_factor_from_saved(blocks, saved, best_f)
    return best_b, float(best_f), float(best_s)


def _gaussian_loo_elpd(Z, y, beta, Sigma, sigma2):
    resid = y - Z @ beta
    var_f = np.einsum("ij,jk,ik->i", Z, Sigma, Z)
    h = np.clip(var_f / max(sigma2, 1e-12), 0.0, 0.999)
    loo_resid = resid / (1.0 - h)
    n = len(y)
    return float(-0.5 * n * np.log(2 * np.pi * sigma2)
                 - 0.5 * np.sum(loo_resid ** 2) / sigma2)


def _logistic_loo_elpd(Z, y, beta, Sigma):
    eta = Z @ beta
    mu = _sigmoid(eta)
    w = np.clip(mu * (1.0 - mu), 1e-10, None)
    var_f = np.clip(np.einsum("ij,jk,ik->i", Z, Sigma, Z), 0.0, None)
    h = np.clip(w * var_f, 0.0, 0.999)
    eta_loo = eta - (y - mu) * var_f / np.clip(1.0 - h, 1e-6, None)
    return float(np.sum(y * eta_loo - np.logaddexp(0.0, eta_loo)))


def _loo_calibrate_gaussian(Z, y, blocks, sigma2, M=None, n_grid=20):
    n, p = Z.shape
    if M is None:
        M = Z.T @ Z
    Zty = Z.T @ y
    Ieye = np.eye(p)
    saved = _save_hypers(blocks)

    def score(fac):
        _apply_factor_from_saved(blocks, saved, fac)
        a = _assemble_precision(blocks)
        H = M / sigma2 + np.diag(a)
        c, low = cho_factor(H, lower=True)
        beta = cho_solve((c, low), Zty / sigma2)
        Sigma = cho_solve((c, low), Ieye)
        return _gaussian_loo_elpd(Z, y, beta, Sigma, sigma2), beta, Sigma

    best_f, best_s = 1.0, -np.inf
    best_b = best_S = None
    for f in np.unique(np.r_[1.0, np.geomspace(1.0, 100.0, n_grid)]):
        s, b, S = score(f)
        if s > best_s:
            best_f, best_s, best_b, best_S = f, s, b, S
    _apply_factor_from_saved(blocks, saved, best_f)
    a = _assemble_precision(blocks)
    H = M / sigma2 + np.diag(a)
    c, low = cho_factor(H, lower=True)
    beta = cho_solve((c, low), Zty / sigma2)
    Sigma = cho_solve((c, low), Ieye)
    return beta, Sigma, float(best_f), float(best_s)


def _loo_calibrate_logistic(Z, y, blocks, beta0, n_grid=20):
    saved = _save_hypers(blocks)

    def score(fac, b0):
        _apply_factor_from_saved(blocks, saved, fac)
        a = _assemble_precision(blocks)
        beta, Sigma, _, _ = _map_logistic(Z, y, a, b0)
        return _logistic_loo_elpd(Z, y, beta, Sigma), beta, Sigma

    best_f, best_s, best_b = 1.0, -np.inf, beta0
    for f in np.unique(np.r_[1.0, np.geomspace(1.0, 100.0, n_grid)]):
        s, b, S = score(f, best_b)
        if s > best_s:
            best_f, best_s, best_b = f, s, b
    _apply_factor_from_saved(blocks, saved, best_f)
    a = _assemble_precision(blocks)
    beta, Sigma, _, _ = _map_logistic(Z, y, a, best_b)
    return beta, Sigma, float(best_f), float(best_s)


# ---------------------------------------------------------------------------
# EM cores
# ---------------------------------------------------------------------------

def _run_gaussian_em(
    Z, y, blocks, M, Zty, Ieye,
    pool_strength, null_pool_strength, a_floor, mackay, mackay_mix,
    max_iter, tol, verbose, update_wiggly=True, sigma2=None,
):
    n, p = Z.shape
    if sigma2 is None:
        sigma2 = max(float(np.var(y)), 1e-12)
    history, prev = [], -np.inf
    lam_bar = kap_bar = 1.0
    beta = np.zeros(p)
    Sigma = Ieye.copy()

    for it in range(max_iter):
        a = _assemble_precision(blocks)
        H = M / sigma2 + np.diag(a)
        c, low = cho_factor(H, lower=True)
        beta = cho_solve((c, low), Zty / sigma2)
        Sigma = cho_solve((c, low), Ieye)
        dSig = np.diag(Sigma).copy()

        resid = y - Z @ beta
        rss = float(resid @ resid)
        logdetH = 2.0 * float(np.sum(np.log(np.diag(c))))
        evidence = (
            -0.5 * n * np.log(2 * np.pi * sigma2)
            - 0.5 * rss / sigma2
            - 0.5 * float(beta @ (a * beta))
            + 0.5 * float(np.sum(np.log(a)))
            - 0.5 * logdetH
        )
        history.append(evidence)

        saved_lams = None if update_wiggly else {id(b): b.lam for b in blocks}
        lam_bar, kap_bar = _em_update_pooled(
            blocks, beta, dSig, a,
            pool_strength if update_wiggly else 0.0,
            null_pool_strength, a_floor,
            mackay=mackay, mackay_mix=mackay_mix,
        )
        if saved_lams is not None:
            for b in blocks:
                if id(b) in saved_lams and b.pen.any():
                    b.lam = saved_lams[id(b)]

        trZSZ = float(np.sum(Sigma * M))
        sigma2 = max((rss + trZSZ) / n, 1e-12)
        if verbose:
            print(f"  gauss iter {it:3d}  ev {evidence:.4f}  s2 {sigma2:.5f}")
        if it > 2 and abs(evidence - prev) < tol * (1 + abs(prev)):
            break
        prev = evidence

    return beta, Sigma, sigma2, history, lam_bar, kap_bar, it + 1


def _run_logistic_em(
    Z, y, blocks,
    pool_strength, null_pool_strength, a_floor, mackay, mackay_mix,
    max_iter, tol, verbose, update_wiggly=True, beta0=None,
):
    n, p = Z.shape
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    history, prev = [], -np.inf
    lam_bar = kap_bar = 1.0
    Sigma = np.eye(p)

    for it in range(max_iter):
        a = _assemble_precision(blocks)
        beta, Sigma, ll, logdetH = _map_logistic(Z, y, a, beta)
        evidence = (
            ll - 0.5 * float(beta @ (a * beta))
            + 0.5 * float(np.sum(np.log(a))) - 0.5 * logdetH
        )
        history.append(evidence)
        dSig = np.diag(Sigma).copy()

        saved_lams = None if update_wiggly else {id(b): b.lam for b in blocks}
        lam_bar, kap_bar = _em_update_pooled(
            blocks, beta, dSig, a,
            pool_strength if update_wiggly else 0.0,
            null_pool_strength, a_floor,
            mackay=mackay, mackay_mix=mackay_mix,
        )
        if saved_lams is not None:
            for b in blocks:
                if id(b) in saved_lams and b.pen.any():
                    b.lam = saved_lams[id(b)]

        if verbose:
            print(f"  logit iter {it:3d}  ev {evidence:.4f}")
        if it > 2 and abs(evidence - prev) < tol * (1 + abs(prev)):
            break
        prev = evidence

    return beta, Sigma, history, lam_bar, kap_bar, it + 1


# ---------------------------------------------------------------------------
# Public fit engines
# ---------------------------------------------------------------------------

def fit_gaussian_eb_pooled(
    Z, y, blocks,
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
    max_iter=200,
    tol=1e-5,
    verbose=False,
    random_state=0,
):
    Z = np.asarray(Z, float)
    y = np.asarray(y, float)
    n, p = Z.shape
    M = Z.T @ Z
    Zty = Z.T @ y
    Ieye = np.eye(p)
    a_floor = float(floor_scale) / max(n, 1) if floor_scale and floor_scale > 0 else 0.0
    budget = _resolve_edf_budget(max_total_edf, n, task="gaussian")
    n_null = float(null_pool_strength)

    history_all, n_iter_total = [], 0
    cal_factor, edf_factor = 1.0, 1.0
    cal_score, n_pruned = None, 0
    sigma2 = max(float(np.var(y)), 1e-12)
    lam_bar = kap_bar = 1.0

    if linear_first:
        _freeze_wiggly(blocks, 1e8)
        beta, Sigma, sigma2, hist, lam_bar, kap_bar, nit = _run_gaussian_em(
            Z, y, blocks, M, Zty, Ieye,
            pool_strength, n_null, a_floor, mackay, mackay_mix,
            max_iter=max(20, max_iter // 3), tol=tol, verbose=verbose,
            update_wiggly=False, sigma2=sigma2,
        )
        history_all.extend(hist)
        n_iter_total += nit
        for b in blocks:
            if not isinstance(b, _InterceptSpec) and b.pen.any():
                b.lam = 1.0

    beta, Sigma, sigma2, hist, lam_bar, kap_bar, nit = _run_gaussian_em(
        Z, y, blocks, M, Zty, Ieye,
        pool_strength, n_null, a_floor, mackay, mackay_mix,
        max_iter=max_iter, tol=tol, verbose=verbose,
        update_wiggly=True, sigma2=sigma2,
    )
    history_all.extend(hist)
    n_iter_total += nit

    # Soft prune near-zero blocks, then one posterior refresh.
    a = _assemble_precision(blocks)
    total_edf, edf = _total_edf(blocks, a, np.diag(Sigma))
    n_pruned = _prune_small_blocks(blocks, edf, prune_edf)
    if n_pruned:
        a = _assemble_precision(blocks)
        H = M / sigma2 + np.diag(a)
        c, low = cho_factor(H, lower=True)
        beta = cho_solve((c, low), Zty / sigma2)
        Sigma = cho_solve((c, low), Ieye)
        total_edf, edf = _total_edf(blocks, a, np.diag(Sigma))

    if budget is not None and total_edf > budget:
        beta, Sigma, total_edf, edf, edf_factor = _enforce_edf_budget_gaussian(
            Z, y, blocks, sigma2, budget, M=M
        )
        resid = y - Z @ beta
        rss = float(resid @ resid)
        sigma2 = max((rss + float(np.sum(Sigma * M))) / n, 1e-12)

    if holdout_calibrate:
        cal_factor, cal_score = _holdout_calibrate_gaussian(
            Z, y, blocks, sigma2, frac=holdout_frac, seed=random_state
        )
    elif loo_calibrate:
        beta, Sigma, cal_factor, cal_score = _loo_calibrate_gaussian(
            Z, y, blocks, sigma2, M=M
        )

    a = _assemble_precision(blocks)
    H = M / sigma2 + np.diag(a)
    c, low = cho_factor(H, lower=True)
    beta = cho_solve((c, low), Zty / sigma2)
    Sigma = cho_solve((c, low), Ieye)
    resid = y - Z @ beta
    rss = float(resid @ resid)
    logdetH = 2.0 * float(np.sum(np.log(np.diag(c))))
    evidence = (
        -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * rss / sigma2
        - 0.5 * float(beta @ (a * beta)) + 0.5 * float(np.sum(np.log(a)))
        - 0.5 * logdetH
    )
    total_edf, edf = _total_edf(blocks, a, np.diag(Sigma))

    return dict(
        beta=beta, Sigma=Sigma, sigma2=sigma2, evidence=evidence,
        history=np.array(history_all), edf=edf, n_iter=n_iter_total,
        total_edf=total_edf, pool_lam_bar=lam_bar, pool_kap_bar=kap_bar,
        edf_scale_factor=edf_factor, cal_scale_factor=cal_factor,
        cal_score=cal_score, loo_scale_factor=cal_factor, loo_elpd=cal_score,
        a_floor=a_floor, edf_budget=budget, n_pruned=n_pruned,
    )


def fit_logistic_eb_pooled(
    Z, y, blocks,
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
    max_iter=100,
    tol=1e-5,
    verbose=False,
    random_state=0,
):
    Z = np.asarray(Z, float)
    y = np.asarray(y, float)
    n, p = Z.shape
    a_floor = float(floor_scale) / max(n, 1) if floor_scale and floor_scale > 0 else 0.0
    budget = _resolve_edf_budget(max_total_edf, n, task="logistic")
    n_null = float(null_pool_strength)

    history_all, n_iter_total = [], 0
    cal_factor, edf_factor = 1.0, 1.0
    cal_score, n_pruned = None, 0
    beta = np.zeros(p)
    lam_bar = kap_bar = 1.0

    if linear_first:
        _freeze_wiggly(blocks, 1e8)
        beta, Sigma, hist, lam_bar, kap_bar, nit = _run_logistic_em(
            Z, y, blocks, pool_strength, n_null, a_floor, mackay, mackay_mix,
            max_iter=max(15, max_iter // 3), tol=tol, verbose=verbose,
            update_wiggly=False, beta0=beta,
        )
        history_all.extend(hist)
        n_iter_total += nit
        for b in blocks:
            if not isinstance(b, _InterceptSpec) and b.pen.any():
                b.lam = 1.0

    beta, Sigma, hist, lam_bar, kap_bar, nit = _run_logistic_em(
        Z, y, blocks, pool_strength, n_null, a_floor, mackay, mackay_mix,
        max_iter=max_iter, tol=tol, verbose=verbose,
        update_wiggly=True, beta0=beta,
    )
    history_all.extend(hist)
    n_iter_total += nit

    a = _assemble_precision(blocks)
    beta, Sigma, ll, logdetH = _map_logistic(Z, y, a, beta)
    total_edf, edf = _total_edf(blocks, a, np.diag(Sigma))
    n_pruned = _prune_small_blocks(blocks, edf, prune_edf)
    if n_pruned:
        a = _assemble_precision(blocks)
        beta, Sigma, ll, logdetH = _map_logistic(Z, y, a, beta)
        total_edf, edf = _total_edf(blocks, a, np.diag(Sigma))

    if budget is not None and total_edf > budget:
        beta, Sigma, total_edf, edf, edf_factor, ll, logdetH = _enforce_edf_budget_logistic(
            Z, y, blocks, budget, beta
        )

    if holdout_calibrate:
        beta, cal_factor, cal_score = _holdout_calibrate_logistic(
            Z, y, blocks, beta, frac=holdout_frac, seed=random_state
        )
    elif loo_calibrate:
        beta, Sigma, cal_factor, cal_score = _loo_calibrate_logistic(
            Z, y, blocks, beta
        )

    a = _assemble_precision(blocks)
    beta, Sigma, ll, logdetH = _map_logistic(Z, y, a, beta)
    evidence = (
        ll - 0.5 * float(beta @ (a * beta))
        + 0.5 * float(np.sum(np.log(a))) - 0.5 * logdetH
    )
    total_edf, edf = _total_edf(blocks, a, np.diag(Sigma))

    return dict(
        beta=beta, Sigma=Sigma, evidence=evidence,
        history=np.array(history_all), edf=edf, n_iter=n_iter_total,
        total_edf=total_edf, pool_lam_bar=lam_bar, pool_kap_bar=kap_bar,
        edf_scale_factor=edf_factor, cal_scale_factor=cal_factor,
        cal_score=cal_score, loo_scale_factor=cal_factor, loo_elpd=cal_score,
        a_floor=a_floor, edf_budget=budget, n_pruned=n_pruned,
    )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

class _PooledMixin:
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

    def _pool_kwargs(self):
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

    def _store_pool_diagnostics(self, fit):
        self.total_edf_ = fit.get("total_edf")
        self.pool_lam_bar_ = fit.get("pool_lam_bar")
        self.pool_kap_bar_ = fit.get("pool_kap_bar")
        self.edf_scale_factor_ = fit.get("edf_scale_factor", 1.0)
        self.cal_scale_factor_ = fit.get("cal_scale_factor", 1.0)
        self.loo_scale_factor_ = fit.get("loo_scale_factor", 1.0)
        self.cal_score_ = fit.get("cal_score")
        self.loo_elpd_ = fit.get("loo_elpd")
        self.a_floor_ = fit.get("a_floor", 0.0)
        self.edf_budget_ = fit.get("edf_budget")
        self.n_pruned_ = fit.get("n_pruned", 0)


class SRAERegressorPooled(_PooledMixin, SRAERegressor):
    """SRAE regression with the pooled anti-overfitting stack.

    Uses the same model family, block constructors, and screening algorithm and
    settings as :class:`~srae.SRAERegressor`, but estimates the block
    hyperparameters differently. Because its main-effect fit supplies the
    residuals used for screening, it need not retain the same interaction set
    as the Type-II estimator. Five devices are layered on top of the Type-II
    fit, each of which can only *increase* regularization:

    1. **Conservative partial pooling** of :math:`\\lambda_j` and
       :math:`\\kappa_j` toward a common scale, taken as
       :math:`\\max(\\lambda^{\\mathrm{MLE}}, \\lambda^{\\mathrm{pool}})`.
    2. **Precision floors** at ``floor_scale / n``.
    3. **Soft pruning** of blocks whose edf falls below ``prune_edf``.
    4. A **total-edf budget** enforced by bisection on a global multiplier.
    5. **Holdout calibration** of a final global multiplier against an
       internal random split.

    Intended for small-:math:`n` / high-:math:`p` problems where empirical
    Bayes is over-confident about its own hyperparameters.

    Read more in the :ref:`User Guide <axis_regularization>`.

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
        Maximum EM iterations.

    tol : float, default=1e-5
        Relative evidence tolerance for the stopping rule.

    feature_names : list of str, default=None
        Component labels. Never modified by ``fit``.

    verbose : bool, default=False
        Print the evidence trace during fitting.

    pool_strength : float, default=2.0
        Pooling weight :math:`\\tau` on the penalized (roughness) precisions.
        Zero disables pooling; larger values pull harder toward the common
        scale.

    null_pool_strength : float, default=2.0
        Pooling weight for the null-space precisions :math:`\\kappa_j`.

    floor_scale : float, default=1.0
        Precision floor, applied as ``floor_scale / n``. Zero disables it.

    max_total_edf : float, 'auto' or None, default='auto'
        Cap on total effective degrees of freedom. ``'auto'`` uses
        :math:`\\max(6, \\min(0.85\\sqrt{n}, n / (2 \\log n)))` for
        regression. ``None`` disables the budget.

    prune_edf : float, default=0.15
        Blocks with edf below this are pinned to a very large precision.
        Zero or ``None`` disables pruning.

    mackay : bool, default=False
        Blend MacKay :math:`\\gamma`-style updates into the M-step. Provided
        for ablation.

    mackay_mix : float, default=0.35
        Blend weight when ``mackay=True``.

    linear_first : bool, default=False
        Run a first pass with all roughness precisions frozen high, so
        low-dimensional null-space structure is established before penalized
        curvature is admitted.

    holdout_calibrate : bool, default=True
        Choose a final global precision multiplier by maximizing held-out
        log predictive density on an internal split. Continuous regression
        targets trigger the calibration helper's random-holdout fallback. The
        score uses :math:`z^\\top\\Sigma z + \\sigma^2`; it does not include
        the public predictor's additional intercept-sampling term
        :math:`\\sigma^2/n`.

        .. warning::

           This is **not** held-out validation. Per-block scales, pruning, the
           edf budget, and ``sigma2`` are estimated from all rows; only the
           scalar multiplier is scored on the internal split. The multiplier
           from the main-effects fit changes the residuals used for interaction
           screening, and calibration runs again after a selected interaction
           refit. Treat the result as a regularized point estimate, not an
           unbiased performance estimate, and use a genuine outer split --
           seeded independently of ``random_state`` -- to measure
           generalization.

    holdout_frac : float, default=0.25
        Fraction of training rows held out for that calibration.

    loo_calibrate : bool, default=False
        Use an approximate leave-one-out criterion instead. Ignored unless
        ``holdout_calibrate=False``.

    random_state : int, default=0
        Seed for the internal calibration split.

    Attributes
    ----------
    total_edf_ : float
        Realized total effective degrees of freedom after all adjustments.

    edf_budget_ : float or None
        The resolved cap that ``max_total_edf`` produced.

    edf_scale_factor_ : float
        Global multiplier applied to meet the budget.

    cal_scale_factor_ : float
        Global multiplier chosen by holdout calibration.

    cal_score_ : float or None
        Held-out log-likelihood at the selected multiplier.

    n_pruned_ : int
        Number of blocks pinned by soft pruning.

    a_floor_ : float
        The applied precision floor.

    pool_lam_bar_, pool_kap_bar_ : float
        Common scales that pooling shrank toward.

    See Also
    --------
    SRAERegressor : Type-II MLE without the pooled stack.
    SRAERegressorSIPooled : Pooled stack plus scale integration.

    Notes
    -----
    The capacity cap is aggressive. On one :math:`n = 80` synthetic design the
    Type-II fit reached :math:`R^2 \\approx 0.99` while the pooled fit reached
    :math:`\\approx 0.53`, despite both selecting the same correct
    interaction. Verify against :class:`~srae.SRAERegressor` rather than
    assuming the pooled variant is uniformly safer.

    ``evidence_`` is *not* comparable with the Type-II path. On a fixed design,
    pooling moves away from the unpooled evidence optimum; screening may also
    produce a different design. Compare variants by held-out predictive score
    only.

    The calibration holdout is drawn using ``random_state``. An outer
    model-selection loop reusing the same splitting scheme would score this
    estimator on rows its calibration already saw.
    """

    def _fit_engine(self, Z, y):
        self._ymean = float(np.mean(y))
        self._n_train = int(len(y))
        return fit_gaussian_eb_pooled(
            Z, y - self._ymean, self.specs_,
            max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
            **self._pool_kwargs(),
        )

    def _finalize(self, y, fit):
        super()._finalize(y, fit)
        self._store_pool_diagnostics(fit)


class SRAEClassifierPooled(_PooledMixin, SRAEClassifier):
    """SRAE classification with the pooled anti-overfitting stack.

    Uses the same model family, prediction link, block constructors, and
    screening algorithm and settings as :class:`~srae.SRAEClassifier`, but
    estimates the block hyperparameters differently. Its changed main-effect
    fit can change the residuals and therefore the selected interaction set.
    See :class:`SRAERegressorPooled` for the five devices applied, and the
    :ref:`User Guide <axis_regularization>` for the mathematics.

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
        Maximum outer iterations; internally capped at 100 for the logistic
        engine.

    tol : float, default=1e-5
        Relative evidence tolerance for the stopping rule.

    feature_names : list of str, default=None
        Component labels. Never modified by ``fit``.

    verbose : bool, default=False
        Print the evidence trace during fitting.

    pool_strength : float, default=2.0
        Pooling weight on the penalized (roughness) precisions.

    null_pool_strength : float, default=2.0
        Pooling weight on the null-space precisions.

    floor_scale : float, default=1.0
        Precision floor, applied as ``floor_scale / n``.

    max_total_edf : float, 'auto' or None, default='auto'
        Cap on total effective degrees of freedom. ``'auto'`` uses
        :math:`\\max(4, \\min(0.50\\sqrt{n}, n / (3.5 \\log n)))` for
        classification -- more aggressive than the regression budget.

    prune_edf : float, default=0.15
        Blocks with edf below this are pinned to a very large precision.

    mackay : bool, default=False
        Blend MacKay :math:`\\gamma`-style updates into the M-step.

    mackay_mix : float, default=0.35
        Blend weight when ``mackay=True``.

    linear_first : bool, default=False
        Run a null-space-only first pass before admitting penalized curvature.

    holdout_calibrate : bool, default=True
        Choose a final global precision multiplier by maximizing held-out
        log predictive density on an internal class-stratified split. The score
        uses moderated binary probabilities. For multiclass, calibration runs
        separately inside each one-vs-rest sub-model; the parent prediction
        link then couples the resulting logits through a softmax.

        .. warning::

           This is **not** held-out validation. Per-block scales, pruning and
           the edf budget are estimated from all rows; only the scalar
           multiplier is scored on the internal split. The multiplier from
           the main-effects fit changes the residuals used for interaction
           screening, and calibration runs again after a selected interaction
           refit. Treat the result as a regularized point estimate, not an
           unbiased performance estimate, and use a genuine outer split --
           seeded independently of ``random_state`` -- to measure
           generalization.

    holdout_frac : float, default=0.25
        Fraction of training rows held out for that calibration.

    loo_calibrate : bool, default=False
        Use an approximate leave-one-out criterion instead.

    random_state : int, default=0
        Seed for the internal calibration split.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    estimators_ : list or None
        One-vs-rest sub-models for multiclass; ``None`` for binary.

    total_edf_ : float
        Realized total effective degrees of freedom. For multiclass, summed
        over sub-models.

    edf_budget_ : float or None
        The resolved cap. Binary models only; for multiclass inspect each
        object in ``estimators_``.

    cal_scale_factor_ : float
        Calibration multiplier; for multiclass, averaged over sub-models.

    n_pruned_ : int
        Number of blocks pinned by soft pruning. Binary models only; for
        multiclass inspect each object in ``estimators_``.

    See Also
    --------
    SRAEClassifier : Type-II MLE without the pooled stack.
    SRAEClassifierSIPooled : Pooled stack plus scale integration.

    Notes
    -----
    ``evidence_`` is not comparable with the Type-II path; compare by held-out
    predictive score. The classification edf budget is tighter than the
    regression one, so verify against :class:`~srae.SRAEClassifier`.
    """

    def _fit_engine(self, Z, y):
        return fit_logistic_eb_pooled(
            Z, y, self.specs_,
            max_iter=min(self.max_iter, 100), tol=self.tol, verbose=self.verbose,
            **self._pool_kwargs(),
        )

    def _finalize(self, y, fit):
        super()._finalize(y, fit)
        self._store_pool_diagnostics(fit)

    def fit(self, X, y):
        from .model import _BaseSRAE

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
        for c in self.classes_:
            est = SRAEClassifierPooled(
                n_knots=self.n_knots,
                max_linear_card=self.max_linear_card,
                feature_types=self.feature_types,
                interactions=self.interactions,
                max_interactions=self.max_interactions,
                max_screen_pairs=self.max_screen_pairs,
                interaction_gain_threshold=self.interaction_gain_threshold,
                max_iter=self.max_iter,
                tol=self.tol,
                feature_names=list(self.feature_names_),
                verbose=self.verbose,
                **self._pool_kwargs(),
            )
            est.fit(Xa, (y == c).astype(float))
            self.estimators_.append(est)
        self.interactions_ = [
            dict(cls=c, **info)
            for c, est in zip(self.classes_, self.estimators_)
            for info in est.interactions_
        ]
        self.evidence_ = float(sum(e.evidence_ for e in self.estimators_))
        self.total_edf_ = float(
            sum(getattr(e, "total_edf_", 0.0) or 0.0 for e in self.estimators_)
        )
        self.cal_scale_factor_ = float(np.mean([
            getattr(e, "cal_scale_factor_", 1.0) or 1.0 for e in self.estimators_
        ]))
        self.loo_scale_factor_ = self.cal_scale_factor_
        return self
