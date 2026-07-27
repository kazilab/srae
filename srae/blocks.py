"""Feature blocks: differentiable-in-spirit basis expansions with quadratic
smoothness penalties, expressed in the Demmler-Reinsch (DR) parametrization.

Each block maps a raw feature (or feature pair) to a design matrix whose
columns are decorrelated with respect to the penalty: the prior precision of
coefficient i is

    a_i = lambda_j * s_i   if s_i > 0   (penalized / "wiggly" direction)
    a_i = kappa_j          if s_i == 0  (null-space direction)

so the whole model's prior precision matrix is diagonal, and the per-block
hyperparameters (lambda_j, kappa_j) have closed-form EM updates under the
marginal-likelihood (evidence) objective.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh

__all__ = [
    "SplineBlock",
    "LinearBlock",
    "FactorBlock",
    "TensorBlock",
    "make_block",
    "normalize_feature_type",
]

# Canonical names accepted by ``make_block`` / ``feature_types``.
_FEATURE_TYPE_ALIASES = {
    "auto": "auto",
    "linear": "linear",
    "lin": "linear",
    "ordinal": "linear",
    "spline": "spline",
    "smooth": "spline",
    "continuous": "spline",
    "numeric": "spline",
    "factor": "factor",
    "categorical": "factor",
    "nominal": "factor",
    "cat": "factor",
}

_FEATURE_TYPE_CHOICES = ("auto", "linear", "spline", "factor")


def normalize_feature_type(feature_type):
    """Map a user-facing feature type label to a canonical name.

    Parameters
    ----------
    feature_type : str or None
        One of ``'auto'``, ``'linear'``, ``'spline'``, ``'factor'``, or a
        documented alias (``'ordinal'``, ``'categorical'``, ``'nominal'``,
        ``'continuous'``, ...). ``None`` is treated as ``'auto'``.

    Returns
    -------
    str
        Canonical name in ``{'auto', 'linear', 'spline', 'factor'}``.

    Raises
    ------
    ValueError
        If ``feature_type`` is not recognized.
    """
    if feature_type is None:
        return "auto"
    key = str(feature_type).lower().strip()
    if key not in _FEATURE_TYPE_ALIASES:
        raise ValueError(
            f"Unknown feature type {feature_type!r}; expected one of "
            f"{list(_FEATURE_TYPE_CHOICES)} "
            f"(aliases: ordinal→linear, categorical/nominal/cat→factor, "
            f"continuous/numeric/smooth→spline)"
        )
    return _FEATURE_TYPE_ALIASES[key]


def _bspline_design(x, t, degree):
    """Dense B-spline design matrix with clamped (constant) extrapolation."""
    lo, hi = t[degree], t[-degree - 1]
    xc = np.clip(x, lo, hi)
    return BSpline.design_matrix(xc, t, degree).toarray()


def _difference_penalty(n_basis, order=2):
    D = np.diff(np.eye(n_basis), n=order, axis=0)
    return D.T @ D


class SplineBlock:
    """Cubic P-spline block for one continuous feature.

    The second-order coefficient-difference penalty is diagonalized in a
    Demmler-Reinsch basis. Its coefficient-space null space is
    two-dimensional. Centering removes the constant fitted contribution,
    leaving a rank-one trend-like contribution governed by ``kappa_j``.

    Because an eigensolver may return arbitrary mixtures of the two
    zero-eigenvalue vectors before centering, the current representation can
    retain two collinear zero-penalty columns for that one fitted
    contribution. With quantile-spaced knots it is trend-like, but not
    generally exactly linear in raw ``x``.
    """

    kind = "spline"

    def __init__(self, n_knots=10, degree=3):
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        uniq = np.unique(x)
        n_interior = int(min(self.n_knots, max(len(uniq) - 2, 1)))
        probs = np.linspace(0, 1, n_interior + 2)[1:-1]
        interior = np.unique(np.quantile(x, probs))
        span = x.max() - x.min()
        eps = 1e-6 * span if span > 0 else 1e-6
        lo, hi = x.min() - eps, x.max() + eps
        k = self.degree
        self.t_ = np.r_[[lo] * (k + 1), interior, [hi] * (k + 1)]

        B = _bspline_design(x, self.t_, k)
        self.col_mean_ = B.mean(axis=0)
        Bc = B - self.col_mean_
        S = _difference_penalty(B.shape[1], order=2)

        s, U = eigh(S)
        s = np.clip(s, 0.0, None)
        Zt = Bc @ U
        # Drop directions that map to the (near-)zero function on the data,
        # e.g. the constant-coefficient null direction after centering.
        norms = np.linalg.norm(Zt, axis=0)
        keep = norms > 1e-8 * max(norms.max(), 1e-12)
        self.U_ = U[:, keep]
        s = s[keep]
        smax = s.max() if s.size else 1.0
        s[s < 1e-10 * smax] = 0.0
        # Scale columns to unit RMS for well-conditioned hyperparameters.
        Zt = Zt[:, keep]
        scale = Zt.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        # Column scaling Z -> Z/scale implies beta -> beta*scale, so the
        # penalty eigenvalues transform as s -> s / scale**2.
        self.s_ = s / scale**2
        return Zt / scale

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        B = _bspline_design(x, self.t_, self.degree)
        return ((B - self.col_mean_) @ self.U_) / self.scale_


class LinearBlock:
    """Single standardized linear column (used for binary / low-cardinality
    ordinal features).  One null-space direction with ARD precision kappa_j.

    The column is linear in the *numeric codes* provided. That is appropriate
    for binary indicators and ordered levels with roughly linear steps; it is
    the wrong basis for unordered nominal labels or non-monotone effects on a
    few levels (use :class:`FactorBlock` or a spline instead).
    """

    kind = "linear"

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.mean_ = x.mean()
        sd = x.std()
        self.sd_ = sd if sd > 1e-12 else 1.0
        self.s_ = np.array([0.0])
        return ((x - self.mean_) / self.sd_)[:, None]

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        return ((x - self.mean_) / self.sd_)[:, None]


class FactorBlock:
    """Nominal factor via drop-one dummy coding with a shared ARD prior.

    Training levels are the sorted unique values of the feature. The last
    level is the reference (all free dummies zero before centering). The
    remaining :math:`K-1` columns are mean-centered and scaled to unit RMS so
    the shared null-space precision :math:`\\kappa_j` is well conditioned.
    Every direction is unpenalized for roughness (:math:`s_i = 0`); shrinkage
    is pure ARD on the factor as a whole.

    Use this for unordered categorical codes and for low-cardinality features
    whose effect is not linear in the codes (e.g. a U-shape on three levels).
    Values not seen at fit time are mapped to the reference coding (all free
    dummies zero before centering).
    """

    kind = "factor"

    def fit(self, x):
        x = np.asarray(x, dtype=float).ravel()
        finite = x[~np.isnan(x)] if np.isnan(x).any() else x
        levels = np.unique(finite)
        if len(levels) < 2:
            raise ValueError(
                "FactorBlock requires at least 2 distinct levels; "
                f"got {len(levels)}"
            )
        self.levels_ = levels
        # Drop the last (sorted) level as the reference category.
        self.free_levels_ = levels[:-1]
        D = self._raw_dummies(x)
        self.col_mean_ = D.mean(axis=0)
        Dc = D - self.col_mean_
        scale = Dc.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        self.s_ = np.zeros(len(self.free_levels_))
        return Dc / scale

    def _raw_dummies(self, x):
        x = np.asarray(x, dtype=float).ravel()
        free = self.free_levels_
        return (x[:, None] == free[None, :]).astype(float)

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        D = self._raw_dummies(x)
        return (D - self.col_mean_) / self.scale_


class TensorBlock:
    """Purified tensor-product block for a pairwise interaction f_jk.

    Each side of the product uses either a low-resolution quadratic B-spline
    marginal (continuous / linear features) or a full indicator basis over
    category levels (features that were fit with :class:`FactorBlock`).
    Using indicators for nominal sides keeps the surface invariant to how
    categories are numbered.  The tensor columns are residualized (at fit
    time) against an intercept, the block's own marginal bases, and the two
    main-effect blocks, so the interaction carries no main-effect signal.
    Ridge penalty (all directions penalized, s_i = 1) with a single lambda_j.
    """

    kind = "tensor"

    def __init__(self, pair, n_knots=2, degree=2, levels=None):
        self.pair = pair
        self.n_knots = n_knots
        self.degree = degree
        # Per-side factor levels. ``None`` means "continuous": use a spline
        # marginal. A level array means the side is nominal, so the marginal
        # must be an indicator basis -- a spline over category codes would
        # make the fitted surface depend on how the levels happen to be
        # numbered, and relabelling an eight-level factor moved test R^2 from
        # 0.34 to 0.10 before this existed.
        self.levels = tuple(levels) if levels is not None else (None, None)

    def _marginal(self, x, fit, side=0):
        lv = self.levels[side] if side < len(self.levels) else None
        if lv is not None:
            # Full indicator basis (every level, not drop-one): purification
            # removes the main effects afterwards, and keeping all columns
            # makes the basis exactly permutation-invariant.
            return (x[:, None] == np.asarray(lv, dtype=float)[None, :]).astype(float)
        k = self.degree
        if fit:
            probs = np.linspace(0, 1, self.n_knots + 2)[1:-1]
            interior = np.unique(np.quantile(x, probs))
            span = x.max() - x.min()
            eps = 1e-6 * span if span > 0 else 1e-6
            t = np.r_[[x.min() - eps] * (k + 1), interior, [x.max() + eps] * (k + 1)]
            self._ts.append(t)
        else:
            t = self._ts[self._ti]
            self._ti += 1
        return _bspline_design(x, t, k)

    @staticmethod
    def _purify_design(Bj, Bk, main_design):
        """Basis the tensor columns are residualized against.

        Includes the tensor's *own* marginal bases, not just the main-effect
        blocks. The two need not span the same space -- a
        :class:`LinearBlock` contributes one column while a continuous
        marginal ``Bj`` is a spline basis, and a :class:`FactorBlock`
        contributes drop-one dummies while its tensor marginal is a *full*
        indicator basis -- and any direction present in ``Bj`` but absent
        from ``main_design`` would otherwise let a pure main effect survive
        purification and be attributed to the pair. Projecting out
        ``[Bj, Bk]`` enforces the functional-ANOVA constraint directly: the
        interaction carries no main effect expressible in its own marginals,
        whatever the main blocks happen to be.

        ``P`` is deliberately rank-deficient (spline marginals partition
        unity; factor indicators plus an intercept are collinear); ``lstsq``
        returns the least-norm solution and the projection itself is unique.
        """
        return np.column_stack([np.ones(len(Bj)), Bj, Bk, main_design])

    def fit(self, xj, xk, main_design):
        """main_design: (n, m) matrix of the two main-effect blocks' columns
        (already transformed), used for purification."""
        self._ts = []
        Bj = self._marginal(np.asarray(xj, float), fit=True, side=0)
        Bk = self._marginal(np.asarray(xk, float), fit=True, side=1)
        T = np.einsum("ij,ik->ijk", Bj, Bk).reshape(len(Bj), -1)
        P = self._purify_design(Bj, Bk, main_design)
        coef, *_ = np.linalg.lstsq(P, T, rcond=None)
        self.purify_coef_ = coef
        Tp = T - P @ coef
        scale = Tp.std(axis=0)
        good = scale > 1e-10
        self.good_ = good
        scale = scale[good]
        self.scale_ = scale
        self.s_ = np.ones(good.sum())
        return Tp[:, good] / scale

    def transform(self, xj, xk, main_design):
        self._ti = 0
        Bj = self._marginal(np.asarray(xj, float), fit=False, side=0)
        Bk = self._marginal(np.asarray(xk, float), fit=False, side=1)
        T = np.einsum("ij,ik->ijk", Bj, Bk).reshape(len(Bj), -1)
        P = self._purify_design(Bj, Bk, main_design)
        Tp = T - P @ self.purify_coef_
        return Tp[:, self.good_] / self.scale_


def make_block(x, n_knots=10, max_linear_card=5, feature_type="auto"):
    """Choose a block type for a single feature.

    Parameters
    ----------
    x : array-like
        Feature column (used only for the ``'auto'`` cardinality rule and to
        validate ``'factor'`` later at ``fit``).
    n_knots : int, default=10
        Interior knots when a :class:`SplineBlock` is chosen.
    max_linear_card : int, default=5
        Under ``feature_type='auto'``, features with at most this many
        distinct values become a :class:`LinearBlock`; set to ``0`` to always
        prefer a spline under auto.
    feature_type : {'auto', 'linear', 'spline', 'factor'} or alias, default='auto'
        Explicit block choice. ``'auto'`` applies the cardinality heuristic.
        ``'factor'`` builds a :class:`FactorBlock` (nominal / non-linear discrete).
        See :func:`normalize_feature_type` for accepted aliases.

    Returns
    -------
    block : SplineBlock or LinearBlock or FactorBlock
        Unfitted block instance.
    """
    ft = normalize_feature_type(feature_type)
    if ft == "linear":
        return LinearBlock()
    if ft == "spline":
        return SplineBlock(n_knots=n_knots)
    if ft == "factor":
        return FactorBlock()
    # auto: low-cardinality → linear (ordinal assumption), else spline
    uniq = np.unique(x[~np.isnan(x)] if np.isnan(x).any() else x)
    if len(uniq) <= max_linear_card:
        return LinearBlock()
    return SplineBlock(n_knots=n_knots)
