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
from scipy.linalg import cho_factor, cho_solve

__all__ = ["BlockSpec", "fit_gaussian_eb", "fit_logistic_eb"]

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
