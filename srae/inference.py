"""Empirical-Bayes inference engines.

Model:  y = Z beta + eps  (Gaussian)   or   y ~ Bernoulli(sigmoid(Z beta))
Prior:  beta_i ~ N(0, 1 / a_i),  a_i = lambda_j s_i (penalized dirs)
                                  a_i = kappa_j     (null dirs)

For the Gaussian model, hyperparameters (lambda_j, kappa_j, sigma^2) are
optimized by EM on the log marginal likelihood. Each M-step has closed form:

    lambda_j <- r_j / sum_i s_i (beta_i^2 + Sigma_ii)      [penalized dirs]
    kappa_j  <- m_j / sum_i (beta_i^2 + Sigma_ii)          [null dirs]
    sigma^2  <- (||y - Z beta||^2 + tr(Z Sigma Z')) / n

The logistic engine uses a Laplace approximation and analogous posterior-moment
fixed-point updates. These procedures estimate continuous shrinkage parameters
without a cross-validated grid, but they do not determine structural choices
such as basis resolution or interaction-screening thresholds.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

__all__ = ["BlockSpec", "fit_gaussian_eb", "fit_logistic_eb",
           "fit_multinomial_eb"]

_A_MIN, _A_MAX = 1e-10, 1e12


class BlockSpec:
    """Bookkeeping for one block inside the stacked design matrix."""

    def __init__(self, name, sl, s):
        self.name = name
        self.sl = sl                      # slice into columns of Z
        self.s = np.asarray(s, float)     # penalty eigenvalues (0 => null dir)
        self.pen = self.s > 0
        self.lam = 1.0
        self.kap = 1.0

    def precisions(self):
        a = np.where(self.pen, self.lam * self.s, self.kap)
        return np.clip(a, _A_MIN, _A_MAX)

    def em_update(self, beta_b, diagSigma_b):
        e2 = beta_b**2 + diagSigma_b
        if self.pen.any():
            denom = float(np.sum(self.s[self.pen] * e2[self.pen]))
            self.lam = np.clip(self.pen.sum() / max(denom, 1e-300), _A_MIN, _A_MAX)
        if (~self.pen).any():
            denom = float(np.sum(e2[~self.pen]))
            self.kap = np.clip((~self.pen).sum() / max(denom, 1e-300), _A_MIN, _A_MAX)

    def edf(self, a, diagSigma_b, scale=1.0):
        """Effective degrees of freedom: tr(I - A Sigma) over the block.
        For the Gaussian case Sigma passed in is the posterior covariance."""
        return float(np.sum(1.0 - a * diagSigma_b / scale))


def _assemble_precision(blocks):
    return np.concatenate([b.precisions() for b in blocks])


def fit_gaussian_eb(Z, y, blocks, max_iter=200, tol=1e-5, verbose=False):
    """EM / evidence maximization for the Gaussian additive model.

    Returns dict with beta, Sigma, sigma2, evidence trace, per-block edf.
    """
    Z = np.asarray(Z, float)
    y = np.asarray(y, float)
    n, p = Z.shape
    M = Z.T @ Z
    Zty = Z.T @ y
    yty = float(y @ y)
    sigma2 = max(float(np.var(y)), 1e-12)
    Ieye = np.eye(p)

    history = []
    prev = -np.inf
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

        # EM updates
        for b in blocks:
            b.em_update(beta[b.sl], dSig[b.sl])
        trZSZ = float(np.sum(Sigma * M))
        sigma2 = max((rss + trZSZ) / n, 1e-12)

        if verbose:
            print(f"iter {it:3d}  evidence {evidence:.4f}  sigma2 {sigma2:.5f}")
        if it > 2 and abs(evidence - prev) < tol * (1 + abs(prev)):
            break
        prev = evidence

    # Refresh the posterior once using the final M-step hyperparameters so the
    # returned coefficients, covariance, sigma2, evidence, and edf are
    # synchronized.  Without this refresh, beta/Sigma correspond to the
    # previous hyperparameter values.
    a = _assemble_precision(blocks)
    H = M / sigma2 + np.diag(a)
    c, low = cho_factor(H, lower=True)
    beta = cho_solve((c, low), Zty / sigma2)
    Sigma = cho_solve((c, low), Ieye)
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
    edf = {b.name: b.edf(a[b.sl], np.diag(Sigma)[b.sl]) for b in blocks}
    return dict(
        beta=beta, Sigma=Sigma, sigma2=sigma2, evidence=evidence,
        history=np.array(history), edf=edf, n_iter=it + 1,
    )


def _sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _map_logistic(Z, y, a, beta0, max_newton=50, tol=1e-8):
    """Penalized MAP estimate via damped Newton (IRLS)."""
    beta = beta0.copy()
    p_dim = Z.shape[1]
    Ieye = np.eye(p_dim)

    def objective(b):
        eta = Z @ b
        # log-lik: sum y*eta - log(1+exp(eta)), stable form
        ll = float(y @ eta - np.sum(np.logaddexp(0.0, eta)))
        return ll - 0.5 * float(b @ (a * b))

    obj = objective(beta)
    for _ in range(max_newton):
        eta = Z @ beta
        mu = _sigmoid(eta)
        w = np.clip(mu * (1 - mu), 1e-10, None)
        grad = Z.T @ (y - mu) - a * beta
        H = (Z.T * w) @ Z + np.diag(a)
        c, low = cho_factor(H, lower=True)
        step = cho_solve((c, low), grad)
        t = 1.0
        for _ls in range(30):
            cand = beta + t * step
            cobj = objective(cand)
            if cobj >= obj - 1e-12:
                break
            t *= 0.5
        beta, new_obj = cand, cobj
        if abs(new_obj - obj) < tol * (1 + abs(obj)):
            obj = new_obj
            break
        obj = new_obj

    eta = Z @ beta
    mu = _sigmoid(eta)
    w = np.clip(mu * (1 - mu), 1e-10, None)
    H = (Z.T * w) @ Z + np.diag(a)
    c, low = cho_factor(H, lower=True)
    Sigma = cho_solve((c, low), Ieye)
    logdetH = 2.0 * float(np.sum(np.log(np.diag(c))))
    ll = float(y @ eta - np.sum(np.logaddexp(0.0, eta)))
    return beta, Sigma, ll, logdetH


def fit_logistic_eb(Z, y, blocks, max_iter=100, tol=1e-5, verbose=False):
    """EM / Laplace evidence maximization for the logistic additive model."""
    Z = np.asarray(Z, float)
    y = np.asarray(y, float)
    n, p = Z.shape
    beta = np.zeros(p)

    history = []
    prev = -np.inf
    for it in range(max_iter):
        a = _assemble_precision(blocks)
        beta, Sigma, ll, logdetH = _map_logistic(Z, y, a, beta)
        evidence = (
            ll
            - 0.5 * float(beta @ (a * beta))
            + 0.5 * float(np.sum(np.log(a)))
            - 0.5 * logdetH
        )
        history.append(evidence)
        dSig = np.diag(Sigma).copy()
        for b in blocks:
            b.em_update(beta[b.sl], dSig[b.sl])
        if verbose:
            print(f"iter {it:3d}  evidence {evidence:.4f}")
        if it > 2 and abs(evidence - prev) < tol * (1 + abs(prev)):
            break
        prev = evidence

    # Refresh the Laplace posterior once using the final precision updates so
    # all returned quantities correspond to the same hyperparameter values.
    a = _assemble_precision(blocks)
    beta, Sigma, ll, logdetH = _map_logistic(Z, y, a, beta)
    evidence = (
        ll
        - 0.5 * float(beta @ (a * beta))
        + 0.5 * float(np.sum(np.log(a)))
        - 0.5 * logdetH
    )
    history.append(evidence)
    edf = {b.name: b.edf(a[b.sl], np.diag(Sigma)[b.sl]) for b in blocks}
    return dict(
        beta=beta, Sigma=Sigma, evidence=evidence,
        history=np.array(history), edf=edf, n_iter=it + 1,
    )


# --------------------------------------------------------------------------
# Joint multinomial (multiclass) engine
# --------------------------------------------------------------------------

def _contrast_basis(K):
    """Orthonormal basis of the sum-to-zero subspace, shape ``(K, K-1)``.

    Parametrizing the ``K`` class logits as ``beta = C gamma`` with
    ``C' C = I`` and ``C' 1 = 0`` is what makes the prior symmetric in the
    classes.  A reference-class parametrization would instead shrink every
    logit toward whichever class happened to be the reference -- an arbitrary
    choice that moved fitted probabilities by up to 0.68 under a measured
    relabelling.  The sum-to-zero subspace is permutation-invariant, so an
    isotropic prior on ``gamma`` induces a class-symmetric prior on the logits.
    """
    Q, _ = np.linalg.qr(np.eye(K) - np.ones((K, K)) / K)
    C = Q[:, :K - 1]
    C = C - C.mean(axis=0, keepdims=True)   # kill round-off along 1
    Q2, _ = np.linalg.qr(C)
    return Q2[:, :K - 1]


def _softmax_rows(Eta):
    """Row-wise softmax over all ``K`` logits."""
    E = Eta - Eta.max(axis=1, keepdims=True)      # overflow-safe
    P = np.exp(E)
    return P / P.sum(axis=1, keepdims=True)


def _multinomial_hessian(Z, P, C, a_full, n_heads):
    """``Z' W Z + diag(a)`` in the sum-to-zero contrast coordinates.

    Contracting the ``K x K`` multinomial weight blocks
    ``p_k (delta_kl - p_l)`` through ``C`` collapses each block to a single
    weighted Gram with weights

    .. code-block:: text

        w_ab = P @ (C[:, a] * C[:, b])  -  (P @ C[:, a]) * (P @ C[:, b])

    The off-diagonal blocks are exactly what independent one-vs-rest fits
    leave at zero.
    """
    p = Z.shape[1]
    H = np.zeros((n_heads * p, n_heads * p))
    U = P @ C                                     # (n, K-1)
    for a in range(n_heads):
        for b in range(a, n_heads):
            w = P @ (C[:, a] * C[:, b]) - U[:, a] * U[:, b]
            blk = (Z.T * w) @ Z
            H[a * p:(a + 1) * p, b * p:(b + 1) * p] = blk
            if b != a:
                H[b * p:(b + 1) * p, a * p:(a + 1) * p] = blk.T
    H[np.diag_indices_from(H)] += a_full
    return H


def _map_multinomial(Z, Y, C, a_full, beta0, n_heads, max_newton=50, tol=1e-8,
                     want_cov=False):
    """Penalized MAP for the multinomial likelihood via damped Newton."""
    n, p = Z.shape
    beta = beta0.copy()

    def logits(b):
        return Z @ b.reshape(n_heads, p).T @ C.T

    def objective(b):
        P = _softmax_rows(logits(b))
        ll = float(np.sum(Y * np.log(np.clip(P, 1e-300, None))))
        return ll - 0.5 * float(b @ (a_full * b))

    obj = objective(beta)
    cand = beta
    for _ in range(max_newton):
        P = _softmax_rows(logits(beta))
        grad = (C.T @ ((Y - P).T @ Z)).ravel() - a_full * beta
        H = _multinomial_hessian(Z, P, C, a_full, n_heads)
        try:
            c, low = cho_factor(H, lower=True)
            step = cho_solve((c, low), grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        t = 1.0
        for _ls in range(30):
            cand = beta + t * step
            cobj = objective(cand)
            if cobj >= obj - 1e-12:
                break
            t *= 0.5
        beta, new_obj = cand, cobj
        if abs(new_obj - obj) < tol * (1 + abs(obj)):
            obj = new_obj
            break
        obj = new_obj

    P = _softmax_rows(logits(beta))
    H = _multinomial_hessian(Z, P, C, a_full, n_heads)
    c, low = cho_factor(H, lower=True)
    logdetH = 2.0 * float(np.sum(np.log(np.diag(c))))
    ll = float(np.sum(Y * np.log(np.clip(P, 1e-300, None))))
    if not want_cov:
        # The EM step needs only diag(Sigma). Inverting the Cholesky factor and
        # taking column norms costs about m^3/6 against the m^3 of solving
        # against a full identity, which matters here because the joint Hessian
        # is (K-1)p square -- the whole point of the model is that it does not
        # factor into K independent p-dimensional problems.
        Linv = solve_triangular(c, np.eye(H.shape[0]), lower=low)
        return beta, np.sum(Linv**2, axis=0), ll, logdetH
    Sigma = cho_solve((c, low), np.eye(H.shape[0]))
    return beta, Sigma, ll, logdetH


def _em_update_shared(b, B, D, n_heads):
    """M-step for a block whose precision is shared across the heads.

    ``B`` and ``D`` are ``(n_heads, block_dim)``.  Sharing one ``lambda_j`` per
    component across classes keeps ``summary()`` reporting a single precision
    per component, as in the binary case, and is the multiclass analogue of one
    smoothing parameter per smooth term.
    """
    e2 = B**2 + D
    if b.pen.any():
        denom = float(np.sum(b.s[b.pen] * e2[:, b.pen]))
        b.lam = np.clip(n_heads * int(b.pen.sum()) / max(denom, 1e-300),
                        _A_MIN, _A_MAX)
    if (~b.pen).any():
        denom = float(np.sum(e2[:, ~b.pen]))
        b.kap = np.clip(n_heads * int((~b.pen).sum()) / max(denom, 1e-300),
                        _A_MIN, _A_MAX)


def fit_multinomial_eb(Z, Y, blocks, max_iter=100, tol=1e-5, verbose=False):
    """EM / Laplace evidence maximization for a joint multinomial model.

    Fits ``K - 1`` free linear predictors on a shared design with class 0 as
    reference, coupling them through one softmax likelihood.  This is the
    joint posterior that stacked one-vs-rest fits only approximate: the
    Hessian carries the cross-class blocks ``-Z' diag(p_k p_l) Z``, which
    independent binary fits set to zero.

    Parameters
    ----------
    Z : ndarray of shape (n_samples, n_columns)
        Shared design matrix.
    Y : ndarray of shape (n_samples, n_classes)
        One-hot response.
    blocks : list of BlockSpec
        Column blocks; each carries one ``lambda`` and ``kappa`` shared across
        the heads.

    Returns
    -------
    dict
        ``beta`` of shape ``(K-1, p)``, the full ``(K-1)p`` square ``Sigma``,
        ``evidence``, ``history``, per-block ``edf`` summed over heads, and
        ``n_iter``.
    """
    Z = np.asarray(Z, float)
    Y = np.asarray(Y, float)
    n, p = Z.shape
    n_heads = Y.shape[1] - 1
    if n_heads < 1:
        raise ValueError("multinomial engine needs at least 2 classes")
    C = _contrast_basis(Y.shape[1])
    beta = np.zeros(n_heads * p)

    history = []
    prev = -np.inf
    it = 0
    for it in range(max_iter):
        a = _assemble_precision(blocks)
        a_full = np.tile(a, n_heads)
        beta, dSig, ll, logdetH = _map_multinomial(Z, Y, C, a_full, beta,
                                                   n_heads)
        evidence = (
            ll
            - 0.5 * float(beta @ (a_full * beta))
            + 0.5 * float(np.sum(np.log(a_full)))
            - 0.5 * logdetH
        )
        history.append(evidence)
        B = beta.reshape(n_heads, p)
        D = dSig.reshape(n_heads, p)
        for b in blocks:
            _em_update_shared(b, B[:, b.sl], D[:, b.sl], n_heads)
        if verbose:
            print(f"iter {it:3d}  evidence {evidence:.4f}")
        if it > 2 and abs(evidence - prev) < tol * (1 + abs(prev)):
            break
        prev = evidence

    a = _assemble_precision(blocks)
    a_full = np.tile(a, n_heads)
    beta, Sigma, ll, logdetH = _map_multinomial(Z, Y, C, a_full, beta, n_heads,
                                                want_cov=True)
    evidence = (
        ll
        - 0.5 * float(beta @ (a_full * beta))
        + 0.5 * float(np.sum(np.log(a_full)))
        - 0.5 * logdetH
    )
    history.append(evidence)
    D = np.diag(Sigma).reshape(n_heads, p)
    edf = {b.name: float(sum(b.edf(a[b.sl], D[h, b.sl]) for h in range(n_heads)))
           for b in blocks}
    return dict(
        beta=beta.reshape(n_heads, p), Sigma=Sigma, evidence=evidence,
        history=np.array(history), edf=edf, n_iter=it + 1, contrasts=C,
    )
