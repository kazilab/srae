"""Feature blocks: differentiable-in-spirit basis expansions with quadratic
smoothness penalties, expressed in the eigenbasis of the penalty.

Each block maps a raw feature (or feature pair) to a design matrix whose
columns diagonalize the penalty -- note this is *not* the Demmler-Reinsch
parametrization, which jointly diagonalizes the penalty and the design inner
product and leaves ``Z'Z`` diagonal as well; here ``Z'Z`` is dense.  Nothing
depends on that stronger property: the evidence uses a full Cholesky of
``M/sigma^2 + diag(a)`` and the edf is the general ``tr(I - A Sigma)``.  What
the rotation buys is that the prior precision of coefficient i is

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
    """Plain coefficient-difference penalty of Eilers and Marx.

    This is the *uniform-knot* P-spline penalty.  It is retained to reproduce
    pre-0.0.7 fits (``SplineBlock.penalty = "difference"``) and as the
    comparison point in the tests; it is no longer the default, because SRAE
    places knots at empirical quantiles.  See
    :func:`_integral_derivative_penalty`.
    """
    D = np.diff(np.eye(n_basis), n=order, axis=0)
    return D.T @ D


def _derivative_coef_operator(t, degree, order):
    """Map degree-``degree`` B-spline coefficients to derivative coefficients.

    The ``order``-th derivative of a degree-``k`` spline is itself a spline, of
    degree ``k - order`` on the trimmed knot vector, whose coefficients are a
    fixed linear function of the original ones.  Returns that operator together
    with the derivative's knots and degree.
    """
    n_basis = len(t) - degree - 1
    cols, t2, k2 = [], None, None
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        d = BSpline(t, c, degree).derivative(order)
        t2, k2 = d.t, d.k
        cols.append(np.asarray(d.c, float)[: len(d.t) - d.k - 1])
    return np.column_stack(cols), t2, k2


def _bspline_gram(t, degree):
    """Exact Gram matrix ``G_ab = integral B_a B_b dx`` of a B-spline basis.

    Products of two degree-``m`` polynomials have degree ``2m``, so a
    Gauss-Legendre rule with ``m + 1`` nodes integrates each knot span
    exactly -- this is a closed form, not an approximation.
    """
    n_basis = len(t) - degree - 1
    G = np.zeros((n_basis, n_basis))
    if n_basis <= 0:
        return G
    gx, gw = np.polynomial.legendre.leggauss(degree + 1)
    lo_edges = t[degree: len(t) - degree - 1]
    hi_edges = t[degree + 1: len(t) - degree]
    for lo, hi in zip(lo_edges, hi_edges):
        if hi <= lo:
            continue
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        pts, w = mid + half * gx, half * gw
        Bm = BSpline.design_matrix(pts, t, degree).toarray()
        G += Bm.T @ (w[:, None] * Bm)
    return G


def _integral_derivative_penalty(t, degree, order=2):
    """Roughness penalty ``Omega_ab = integral B_a^(d) B_b^(d) dx``.

    The quadratic form ``beta' Omega beta`` is exactly
    :math:`\\int (f^{(d)})^2\\,dx` for ``f = B beta``, whatever the knot
    spacing.  The plain difference penalty
    (:func:`_difference_penalty`) coincides with this only for equally spaced
    knots; SRAE places knots at empirical quantiles, where the two differ
    substantially -- on a gamma-distributed feature the difference penalty cost
    about 25 nats of evidence and inflated the reported edf by ~40% relative to
    this one, because it under-penalizes curvature wherever knots are dense.

    Factoring as ``Omega = D' G D`` (derivative operator, then the exact Gram
    matrix of the derivative basis) keeps it well behaved when two quantile
    knots nearly coincide: the entries of ``D`` grow but the corresponding
    interval lengths inside ``G`` shrink, so ``Omega`` stays finite.

    A further benefit is that the null space is exactly the space of degree
    ``< order`` polynomials -- for ``order=2``, the straight lines in raw
    ``x``.  Under the difference penalty the null space was only *trend-like*
    on non-uniform knots, which is what forced the hedged wording around
    ``kappa_j`` before 0.0.7.
    """
    n_basis = len(t) - degree - 1
    if n_basis <= order or degree < order:
        return np.zeros((n_basis, n_basis))
    D, t2, k2 = _derivative_coef_operator(t, degree, order)
    Om = D.T @ _bspline_gram(t2, k2) @ D
    return 0.5 * (Om + Om.T)


def _marginal_penalty_parts(t, degree, n_basis, nominal):
    """``(Omega, Omega1, G)`` for one side of a tensor product.

    For a continuous side these are the exact integrals of ``(B'')^2``,
    ``(B')^2`` and ``B^2`` over a knot vector rescaled to ``[0, 1]``.  The
    rescaling is what makes the tensor penalty invariant to the *units* of the
    feature: the three Kronecker terms below carry different powers of the
    domain length, so on raw knots their relative weighting would drift as a
    feature was re-expressed in millimetres rather than metres -- the standard
    non-invariance of an isotropic thin-plate penalty under differential
    scaling of covariates.

    A nominal side has no smoothness to penalize -- categories are unordered --
    so all three parts are the identity.  The tensor penalty then reduces to a
    Sobolev-type penalty applied independently per category, which is
    permutation-invariant in the levels and so preserves the coding invariance
    that the indicator marginals exist to provide.
    """
    if nominal:
        eye = np.eye(n_basis)
        return eye, eye, eye
    span = t[-1] - t[0]
    tn = (t - t[0]) / (span if span > 0 else 1.0)
    return (_integral_derivative_penalty(tn, degree, order=2),
            _integral_derivative_penalty(tn, degree, order=1),
            _bspline_gram(tn, degree))


def _tensor_roughness_penalty(parts_j, parts_k):
    """Isotropic tensor-product roughness penalty.

    ``beta' Omega beta`` is

    .. code-block:: text

        double-integral of  f_jj^2 + 2 f_jk^2 + f_kk^2

    for ``f = sum_uv beta_uv B_u(x_j) B_v(x_k)``, the thin-plate-style
    combination of pure and mixed second derivatives.  Written with Kronecker
    products of the one-dimensional pieces it costs three small products.

    Unlike a ridge on the coefficients, this is a functional of ``f`` rather
    than of its coordinates, so under a change of marginal basis
    ``B_j -> B_j R``, ``B_k -> B_k S`` it transforms as
    ``Omega -> (R kron S)' Omega (R kron S)`` while the purified design
    transforms as ``T_p -> T_p (R kron S)``.  The induced prior
    ``T_p Omega^+ T_p'`` is therefore unchanged -- which is exactly the
    invariance the isotropic ridge lacked.

    One ``lambda_jk`` per surface is retained rather than a separate smoothing
    parameter per margin: the anisotropic form would double the hyperparameters
    of every candidate pair during screening for a surface that is discarded
    unless it clears the gain threshold.

    The null space is the affine functions ``a + b x_j + c x_k`` -- note the
    bilinear ``x_j x_k`` is *not* in it, being penalized through the mixed
    term.  Purification has already removed the affine part, so those
    directions arrive at the eigendecomposition as (near-)zero columns and are
    dropped there.
    """
    Om_j, Om1_j, G_j = parts_j
    Om_k, Om1_k, G_k = parts_k
    Om = (np.kron(Om_j, G_k) + 2.0 * np.kron(Om1_j, Om1_k) + np.kron(G_j, Om_k))
    return 0.5 * (Om + Om.T)


class SplineBlock:
    """Cubic penalized-spline block for one continuous feature.

    Roughness is penalized by the exact integrated squared second derivative,
    ``integral (f'')^2 dx``, and the block is rotated into the eigenbasis of
    that penalty so the prior precision is diagonal.  Its coefficient-space
    null space is two-dimensional -- the straight lines in raw ``x``.
    Centering removes the constant fitted contribution, leaving exactly one
    surviving function: a straight line in raw ``x``, governed by ``kappa_j``
    and carried by a single column.

    An eigensolver returns an arbitrary mixture of the two zero-eigenvalue
    vectors, so a naive norm filter would keep two perfectly collinear columns
    for that one function.  Since 0.0.9 the null eigenspace is rotated to
    separate the identified direction from the zero function before the design
    is built, so ``n_coef`` counts distinct functions and ``edf`` is bounded by
    a column count that means something.

    Notes
    -----
    ``penalty`` is a class attribute rather than a constructor parameter, so it
    is deliberately absent from ``get_params`` and does not survive
    :func:`sklearn.base.clone`.  Setting it to ``"difference"`` restores the
    pre-0.0.7 plain coefficient-difference penalty, which is the Eilers-Marx
    construction for *equally spaced* knots; it exists to reproduce results
    published before 0.0.7, not as a quantity to tune.
    """

    kind = "spline"

    #: 'integral' (default) penalizes the exact integrated squared second
    #: derivative. 'difference' restores the pre-0.0.7 uniform-knot penalty.
    penalty = "integral"

    def __init__(self, n_knots=10, degree=3):
        self.n_knots = n_knots
        self.degree = degree

    def _penalty_matrix(self, n_basis):
        if getattr(self, "penalty", "integral") == "difference":
            return _difference_penalty(n_basis, order=2)
        return _integral_derivative_penalty(self.t_, self.degree, order=2)

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
        S = self._penalty_matrix(B.shape[1])

        s, U = eigh(S)
        s = np.clip(s, 0.0, None)
        # Normalize the penalty scale. The roughness penalty carries units of
        # x^-3, so the same feature expressed in millimetres rather than metres
        # shifts every eigenvalue by 1e9. Left alone that strands the EM update
        # for lambda_j -- which starts at 1 -- in a degenerate fixed point: once
        # the prior dominates completely the posterior equals the prior, the
        # update returns lambda_j unchanged, and the component collapses to its
        # null space regardless of the data. Rescaling is a reparameterization
        # of lambda_j alone; the relative eigenvalues carry the roughness
        # weighting and are untouched, so the induced prior family, the fit and
        # the evidence are unchanged, while invariance to the units of x is
        # restored exactly.
        if s.size and s.max() > 0:
            s = s / s.max()
        s[s < 1e-10] = 0.0

        # Canonicalize the penalty null space before it reaches the design.
        #
        # The order-2 roughness penalty annihilates a two-dimensional space --
        # the constant and the straight line -- but centering has already
        # removed the constant *fitted* contribution, so only one function
        # survives. An eigensolver returns an arbitrary mixture of the two
        # zero-eigenvalue vectors, and in practice always one that leaves both
        # columns non-constant, so a naive norm filter keeps two perfectly
        # collinear columns for that single function.
        #
        # That redundancy is inert -- two collinear coordinates sharing an
        # isotropic kappa_j telescope to the same EM fixed point as one, so the
        # evidence, edf and fitted values are identical either way -- but it
        # overstates ``n_coef`` by one and loosens the documented bound that
        # edf cannot exceed the block's column count. Rotating within the null
        # eigenspace to separate the identified direction from the zero
        # function costs nothing and reports honestly. The retained column is
        # exactly a straight line in raw x.
        pen, nul = s > 0.0, s <= 0.0
        U_pen, U_nul = U[:, pen], U[:, nul]
        if U_nul.shape[1]:
            Zn = Bc @ U_nul
            _, dn, Vn = np.linalg.svd(Zn, full_matrices=False)
            U_nul = U_nul @ Vn[dn > 1e-8 * max(dn.max(), 1e-12)].T
        U = np.column_stack([U_pen, U_nul])
        s = np.concatenate([s[pen], np.zeros(U_nul.shape[1])])

        Zt = Bc @ U
        # Drop any remaining direction that maps to the (near-)zero function.
        norms = np.linalg.norm(Zt, axis=0)
        keep = norms > 1e-8 * max(norms.max(), 1e-12)
        self.U_ = U[:, keep]
        s = s[keep]
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

    The surface is penalized by its integrated squared second derivatives
    (:func:`_tensor_roughness_penalty`) with a single ``lambda_jk``, and the
    block is rotated into the eigenbasis of that penalty.  Because the penalty
    is a functional of the fitted surface rather than of its coordinates, the
    screening gain depends on the space the marginals span and not on how they
    are parametrized -- which the isotropic ridge used before 0.0.8 did not
    provide.

    Notes
    -----
    ``penalty`` is a class attribute rather than a constructor parameter, so it
    is absent from ``get_params`` and does not survive
    :func:`sklearn.base.clone`.  Setting it to ``"ridge"`` restores the
    pre-0.0.8 isotropic ridge; it exists to reproduce results published before
    0.0.8, not as a quantity to tune.
    """

    kind = "tensor"

    #: 'roughness' (default) penalizes the integrated squared second
    #: derivatives of the surface. 'ridge' restores the pre-0.0.8 isotropic
    #: ridge on the tensor coefficients, which was not basis-invariant.
    penalty = "roughness"

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

    def _penalty_matrix(self, p, q):
        """Penalty on the raw tensor coefficients, before purification."""
        if getattr(self, "penalty", "roughness") == "ridge":
            return np.eye(p * q)
        parts_j = _marginal_penalty_parts(
            self._ts[0] if self.levels[0] is None else None,
            self.degree, p, self.levels[0] is not None)
        parts_k = _marginal_penalty_parts(
            self._ts[-1] if self.levels[1] is None else None,
            self.degree, q, self.levels[1] is not None)
        return _tensor_roughness_penalty(parts_j, parts_k)

    def fit(self, xj, xk, main_design):
        """main_design: (n, m) matrix of the two main-effect blocks' columns
        (already transformed), used for purification.

        Purification removes ``span[1, Bj, Bk]``, which -- because both spline
        marginals partition unity -- lies *inside* the tensor's own column
        space with dimension ``5 + 5 - 1 = 9``.  The raw 25 columns therefore
        carry design rank 16.

        Three of those nine are the roughness penalty's affine null space, and
        they are dropped here rather than handed to ``kappa_j``: purification
        has already removed the affine part, so they carry the zero function.
        Dropping them makes the induced prior exactly
        ``T_p Omega^+ T_p' / lambda_jk``, which is the quantity satisfying the
        congruence identity that gives basis invariance.  Selecting instead on
        the design's column norms -- which coordinates *look* null -- would
        not be invariant: two bases would disagree about whether a direction
        was dropped or merely unpenalized, and the gain would move.  The
        shipped block therefore has 22 columns of design rank 16.

        The remaining six directions are penalized but unidentified.  They
        contribute nothing to ``edf``, which is bounded by the design rank
        because those directions lie in the null space of the design's Gram,
        and the proper prior keeps the posterior computable.
        """
        self._ts = []
        Bj = self._marginal(np.asarray(xj, float), fit=True, side=0)
        Bk = self._marginal(np.asarray(xk, float), fit=True, side=1)
        T = np.einsum("ij,ik->ijk", Bj, Bk).reshape(len(Bj), -1)
        P = self._purify_design(Bj, Bk, main_design)
        coef, *_ = np.linalg.lstsq(P, T, rcond=None)
        self.purify_coef_ = coef
        Tp = T - P @ coef

        # No column dropping or column scaling before the penalty is applied.
        # Both select coordinates, and the penalty's transformation law
        # ``Omega -> W' Omega W`` only pairs with ``T_p -> T_p W`` for
        # invertible ``W``; a coordinate-dependent projection breaks it, and
        # with it the basis invariance this penalty exists to provide.
        S = self._penalty_matrix(Bj.shape[1], Bk.shape[1])
        s, U = eigh(S)
        s = np.clip(s, 0.0, None)
        # Drop the penalty's null space -- the affine functions -- rather than
        # handing it to kappa_j. Purification has already removed the affine
        # part, so these directions carry the zero function and contribute
        # nothing; dropping them makes the induced prior exactly
        # ``T_p Omega^+ T_p' / lambda``, which is the quantity that satisfies
        # the congruence identity and so is invariant to the marginal basis.
        # Selecting on the design's column norms instead would not be: which
        # columns look null depends on the coordinates, and the two bases then
        # disagree about whether a direction is dropped or merely unpenalized.
        keep = s > 1e-9 * s.max() if s.size and s.max() > 0 else np.ones(len(s), bool)
        self.U_ = U[:, keep]
        s = s[keep]
        if s.size and s.max() > 0:
            s = s / s.max()          # see SplineBlock.fit: units of lambda only
        Zt = (Tp @ U)[:, keep]
        scale = Zt.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        self.s_ = s / scale**2
        return Zt / scale

    def transform(self, xj, xk, main_design):
        self._ti = 0
        Bj = self._marginal(np.asarray(xj, float), fit=False, side=0)
        Bk = self._marginal(np.asarray(xk, float), fit=False, side=1)
        T = np.einsum("ij,ik->ijk", Bj, Bk).reshape(len(Bj), -1)
        P = self._purify_design(Bj, Bk, main_design)
        Tp = T - P @ self.purify_coef_
        return (Tp @ self.U_) / self.scale_


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
