"""Self-Regularizing Additive Estimator (SRAE).

f(x) = beta_0 + sum_j f_j(x_j) + sum_(j,k) f_jk(x_j, x_k)

* Nonlinearity: P-spline main effects + purified tensor-product interactions.
* Overfitting / underfitting: per-component smoothness selected by maximizing
  the marginal likelihood (evidence) -- a single objective balancing fit
  against complexity (Occam's razor).  For components the data supports the
  optimum is interior.  For a component with no signal it is *not*: the ARD
  precision maximizes at infinity, so the evidence approaches a boundary
  limit and the reported precision keeps growing with ``max_iter`` while the
  evidence trace looks flat.  Such components are flagged by
  ``at_boundary_`` on each fitted response; read their precisions as
  "pruned", not as estimates. Multiclass parent classifiers keep these
  per-response diagnostics on their one-vs-rest ``estimators_``.
* No hyperparameter search: all regularization strengths are latent variables
  with closed-form EM updates; interactions are discovered automatically by
  evidence-gain screening.
* Transparency: main effects are plottable 1-D functions with pointwise
  posterior credible bands, while interactions are plottable 2-D mean
  surfaces; complexity allocation is reported as effective degrees of freedom
  (edf) per feature.
"""

from __future__ import annotations

import warnings

import numpy as np

from .blocks import (
    FactorBlock,
    LinearBlock,
    SplineBlock,
    TensorBlock,
    make_block,
    normalize_feature_type,
)
from .inference import (BlockSpec, fit_gaussian_eb, fit_logistic_eb,
                        fit_multinomial_eb, _sigmoid)

# Main-effect blocks that own a single input column (not intercept / tensor).
_MAIN_BLOCKS = (SplineBlock, LinearBlock, FactorBlock)

#: Largest ``(K-1) * n_columns`` for which the joint multinomial refit is
#: attempted. The Hessian is that square and is factorized inside every EM
#: step, so time grows as its cube and memory as its square.
#:
#: 4000 admits realistic problems -- 10 classes over 40 features is 3249, about
#: half a minute -- while excluding the degenerate ones. sklearn's
#: ``check_estimator`` fits a 200-class problem that would ask for a
#: 14129-square Hessian: 1.6 GB per matrix, and enough of them to be killed by
#: the OOM reaper. Raise it only if you have measured the fit you want.
_MAX_JOINT_DIM = 4000

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
except ImportError as _exc:  # pragma: no cover - exercised only without sklearn
    raise ImportError(
        "SRAE estimators require scikit-learn for sklearn-compatible "
        "BaseEstimator / mixins (clone, Pipeline, GridSearchCV, CV, etc.). "
        "Install with: pip install 'srae[sklearn]' or pip install scikit-learn"
    ) from _exc

__all__ = ["SRAERegressor", "SRAEClassifier"]


class _InterceptSpec(BlockSpec):
    """Fixed, essentially-flat prior on the intercept; excluded from EM."""

    def __init__(self, sl):
        super().__init__("(intercept)", sl, np.array([0.0]))
        self.kap = 1e-6

    def em_update(self, beta_b, diagSigma_b):  # noqa: D401 - intentional no-op
        pass


class _BaseSRAE(BaseEstimator):
    """Shared SRAE design / screening logic with sklearn estimator protocol.

    Inherits :class:`~sklearn.base.BaseEstimator` so that ``get_params``,
    ``set_params``, :func:`~sklearn.base.clone`, :class:`~sklearn.pipeline.Pipeline`,
    :class:`~sklearn.model_selection.GridSearchCV`, and
    :func:`~sklearn.model_selection.cross_val_score` work out of the box.
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
    ):
        self.n_knots = n_knots
        self.max_linear_card = max_linear_card
        self.feature_types = feature_types
        self.interactions = interactions
        self.max_interactions = max_interactions
        self.max_screen_pairs = max_screen_pairs
        self.interaction_gain_threshold = interaction_gain_threshold
        self.max_iter = max_iter
        self.tol = tol
        self.feature_names = feature_names
        self.verbose = verbose

    # ---------------------------------------------------------------- utils
    def _coerce(self, X):
        if hasattr(X, "values"):
            X = X.values
        return np.asarray(X, dtype=float)

    def _resolve_names(self, X, p):
        """Fitted component labels: explicit param > DataFrame columns > x0..xp.

        Written to ``feature_names_``, never back to the ``feature_names``
        constructor parameter -- sklearn requires ``fit`` to leave parameters
        untouched, and latching the first dataset's columns there silently
        mislabels every later refit.
        """
        if self.feature_names is not None:
            labels = [str(n) for n in self.feature_names]
        elif hasattr(X, "columns"):
            labels = [str(c) for c in X.columns]
        else:
            labels = [f"x{j}" for j in range(p)]
        if len(labels) != p:
            raise ValueError(
                f"feature_names has {len(labels)} entries but X has {p} columns"
            )
        self.feature_names_ = labels
        return labels

    def _set_sklearn_fit_attrs(self, X):
        """Record sklearn fitted metadata (n_features_in_, feature_names_in_)
        and resolve the fitted component labels (``feature_names_``).

        Idempotent: safe to call twice with the same ``X`` within one ``fit``.
        """
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X_arr.shape}")
        self.n_features_in_ = int(X_arr.shape[1])
        names = None
        if hasattr(X, "columns"):
            names = np.asarray([str(c) for c in X.columns], dtype=object)
        elif self.feature_names is not None:
            names = np.asarray([str(n) for n in self.feature_names], dtype=object)
        if names is not None and len(names) == self.n_features_in_:
            self.feature_names_in_ = names
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_  # stale names from an earlier fit
        self._resolve_names(X, self.n_features_in_)

    def _names(self, p):
        if getattr(self, "feature_names_", None) is None:
            self._resolve_names(None, p)
        return self.feature_names_

    def _resolve_feature_types(self, p, names):
        """Canonical per-column types from the ``feature_types`` parameter.

        Accepted forms
        --------------
        * ``None`` — every column is ``'auto'``.
        * a single string — applied to every column.
        * a sequence of length ``p`` — one type per column.
        * a dict — keys are feature names or integer indices; missing keys
          default to ``'auto'``.

        Writes the resolved list to ``feature_types_`` and returns it.
        """
        ft = self.feature_types
        if ft is None:
            resolved = ["auto"] * p
        elif isinstance(ft, str):
            resolved = [normalize_feature_type(ft)] * p
        elif isinstance(ft, dict):
            resolved = ["auto"] * p
            name_to_j = {n: j for j, n in enumerate(names)}
            for key, val in ft.items():
                if isinstance(key, str):
                    if key not in name_to_j:
                        raise ValueError(
                            f"feature_types key {key!r} is not among "
                            f"feature names {list(names)}"
                        )
                    j = name_to_j[key]
                else:
                    j = int(key)
                    if j < 0 or j >= p:
                        raise ValueError(
                            f"feature_types index {j} is out of range "
                            f"for {p} features"
                        )
                resolved[j] = normalize_feature_type(val)
        else:
            try:
                seq = list(ft)
            except TypeError as exc:
                raise TypeError(
                    "feature_types must be None, a string, a sequence of "
                    f"length n_features, or a dict; got {type(ft).__name__}"
                ) from exc
            if len(seq) != p:
                raise ValueError(
                    f"feature_types has {len(seq)} entries but X has "
                    f"{p} columns"
                )
            resolved = [normalize_feature_type(t) for t in seq]
        self.feature_types_ = resolved
        return resolved

    # -------------------------------------------------------- main effects
    def _fit_main_blocks(self, X):
        n, p = X.shape
        names = self._names(p)
        types = self._resolve_feature_types(p, names)
        self.blocks_ = []      # fitted transformers, aligned with specs
        self.specs_ = []
        cols = []
        start = 0
        if self._needs_intercept:
            cols.append(np.ones((n, 1)))
            self.specs_.append(_InterceptSpec(slice(0, 1)))
            self.blocks_.append(None)
            start = 1
        for j in range(p):
            blk = make_block(
                X[:, j],
                n_knots=self.n_knots,
                max_linear_card=self.max_linear_card,
                feature_type=types[j],
            )
            Zj = blk.fit(X[:, j])
            sl = slice(start, start + Zj.shape[1])
            self.specs_.append(BlockSpec(names[j], sl, blk.s_))
            self.blocks_.append(blk)
            cols.append(Zj)
            start += Zj.shape[1]
        self._main_feature_of_block = {}
        for i, spec in enumerate(self.specs_):
            if isinstance(self.blocks_[i], _MAIN_BLOCKS):
                self._main_feature_of_block[spec.name] = i
        return np.column_stack(cols)

    def _main_design_for_pair(self, Z, pair):
        """Columns of the two main-effect blocks (for tensor purification)."""
        idx = []
        for j in pair:
            spec = self._spec_of_feature(j)
            idx.extend(range(spec.sl.start, spec.sl.stop))
        return Z[:, idx]

    def _pair_levels(self, pair):
        """Per-side factor levels for a tensor block, ``None`` if continuous.

        A nominal feature must contribute an indicator marginal to the tensor,
        otherwise the interaction surface depends on the arbitrary numbering
        of its categories even though ``feature_types='factor'`` made the main
        effect coding-invariant.
        """
        out = []
        for j in pair:
            blk = None
            name = self.feature_names_[j]
            idx = getattr(self, "_main_feature_of_block", {}).get(name)
            if idx is not None:
                blk = self.blocks_[idx]
            out.append(getattr(blk, "levels_", None) if isinstance(blk, FactorBlock)
                       else None)
        return tuple(out)

    def _spec_of_feature(self, j):
        name = self.feature_names_[j]
        for spec in self.specs_:
            if spec.name == name:
                return spec
        raise KeyError(name)

    # -------------------------------------------------------- interactions
    def _screen_interactions(self, X, Z, resid):
        """Rank candidate pairs by evidence gain of a purified tensor block
        fitted to residuals; keep pairs whose gain clears the threshold."""
        n, p = X.shape
        pairs = [(j, k) for j in range(p) for k in range(j + 1, p)]
        if len(pairs) > self.max_screen_pairs:
            # Cheap pre-ranking: |corr| of the standardized product with resid.
            scores = []
            r = resid - resid.mean()
            rn = np.linalg.norm(r) + 1e-300
            for (j, k) in pairs:
                xj = (X[:, j] - X[:, j].mean())
                xk = (X[:, k] - X[:, k].mean())
                prod = xj * xk
                prod -= prod.mean()
                denom = np.linalg.norm(prod) * rn + 1e-300
                scores.append(abs(float(prod @ r)) / denom)
            order = np.argsort(scores)[::-1][: self.max_screen_pairs]
            pairs = [pairs[i] for i in order]

        results = []
        var_r = float(np.var(resid))
        n_obs = len(resid)
        # Evidence of the empty (noise-only) model for the residuals.
        base_ev = -0.5 * n_obs * (np.log(2 * np.pi * max(var_r, 1e-12)) + 1.0)
        for (j, k) in pairs:
            tb = TensorBlock((j, k), levels=self._pair_levels((j, k)))
            try:
                T = tb.fit(X[:, j], X[:, k], self._main_design_for_pair(Z, (j, k)))
            except ValueError as exc:
                # Numerical failure for this pair only (non-finite values, or
                # LinAlgError from the purification lstsq -- a ValueError
                # subclass). Skip the candidate but say so: a silent skip is
                # indistinguishable from a pair that simply screened poorly.
                # Anything else (TypeError, AttributeError, ...) is a bug and
                # must propagate rather than be swallowed as a skipped pair.
                warnings.warn(
                    f"skipping interaction candidate "
                    f"{self.feature_names_[j]}*{self.feature_names_[k]}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if T.shape[1] == 0:
                continue
            spec = BlockSpec(f"{self.feature_names_[j]}*{self.feature_names_[k]}",
                             slice(0, T.shape[1]), tb.s_)
            fit = fit_gaussian_eb(T, resid, [spec], max_iter=30, tol=1e-4)
            gain = fit["evidence"] - base_ev
            results.append((gain, (j, k), tb))
        results.sort(key=lambda t: -t[0])
        chosen = [(g, pr, tb) for g, pr, tb in results
                  if g > self.interaction_gain_threshold][: self.max_interactions]
        return chosen

    def _add_interaction_blocks(self, X, Z, chosen):
        cols = [Z]
        start = Z.shape[1]
        for gain, pair, tb in chosen:
            T = tb.transform(X[:, pair[0]], X[:, pair[1]],
                             self._main_design_for_pair(Z, pair))
            name = f"{self.feature_names_[pair[0]]}*{self.feature_names_[pair[1]]}"
            self.specs_.append(BlockSpec(name, slice(start, start + T.shape[1]), tb.s_))
            self.blocks_.append(tb)
            cols.append(T)
            start += T.shape[1]
        self.interactions_ = [
            dict(pair=pair, name=f"{self.feature_names_[pair[0]]}*{self.feature_names_[pair[1]]}",
                 screen_gain=float(gain))
            for gain, pair, tb in chosen
        ]
        return np.column_stack(cols)

    # ------------------------------------------------------------ design
    def _design(self, X):
        """Full design matrix for new data (after fit)."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        cols = []
        # main effects (+ intercept) in original order
        for blk, spec in zip(self.blocks_, self.specs_):
            if isinstance(spec, _InterceptSpec):
                cols.append(np.ones((n, 1)))
            elif isinstance(blk, _MAIN_BLOCKS):
                j = self.feature_names_.index(spec.name)
                cols.append(blk.transform(X[:, j]))
        Zmain = np.column_stack(cols) if cols else np.empty((n, 0))
        for blk, spec in zip(self.blocks_, self.specs_):
            if isinstance(blk, TensorBlock):
                j, k = blk.pair
                md = self._main_design_for_pair(Zmain, (j, k))
                cols.append(blk.transform(X[:, j], X[:, k], md))
        return np.column_stack(cols)

    # ------------------------------------------------------------- report
    def summary(self):
        """Per-component report: type, size, edf, importance, hyperparameters.

        Returns
        -------
        summary : pandas.DataFrame
            One row per component, sorted by ``importance`` descending, with
            columns:

            ``component``
                Component label; interactions appear as ``"a*b"``.
            ``kind``
                One of ``'spline'``, ``'linear'``, ``'factor'``, ``'tensor'``.
            ``n_coef``
                Number of basis columns in the block.
            ``edf``
                Effective degrees of freedom, :math:`\\sum_i
                (1 - a_i \\Sigma_{ii})`, bounded by ``n_coef``.
            ``importance``
                Standard deviation of the component's contribution to the
                predictor on the training data.
            ``lam``
                Roughness precision :math:`\\lambda_j`; ``NaN`` when the block
                has no penalized direction.
            ``kappa``
                Null-space precision :math:`\\kappa_j`; ``NaN`` when the block
                has no null direction.

        Notes
        -----
        ``importance`` is an in-sample dispersion, not a causal effect size
        and not a permutation importance. Correlated features can trade off
        against one another.

        Read ``edf`` together with ``kind``, ``lam`` and ``kappa``. ``edf``
        near 1 means roughly one effective direction remains, not universally
        a straight line. For a spline with large ``lam`` that direction is a
        straight line in raw ``x``; for factor and tensor blocks it has a
        different meaning.
        ``edf`` near 0 means the component is effectively switched off, while
        spline ``edf`` above 2 supports penalized curvature.
        """
        import pandas as pd

        Z = self._Ztrain
        rows = []
        for spec, blk in zip(self.specs_, self.blocks_):
            if isinstance(spec, _InterceptSpec):
                continue
            contrib = Z[:, spec.sl] @ self.beta_[spec.sl]
            rows.append(dict(
                component=spec.name,
                kind=blk.kind if blk is not None else "intercept",
                n_coef=spec.sl.stop - spec.sl.start,
                edf=round(self.edf_[spec.name], 2),
                importance=float(np.std(contrib)),
                lam=float(spec.lam) if spec.pen.any() else np.nan,
                kappa=float(spec.kap) if (~spec.pen).any() else np.nan,
            ))
        df = pd.DataFrame(rows).sort_values("importance", ascending=False)
        return df.reset_index(drop=True)

    def shape_function(self, feature, grid=None, n_grid=200):
        """Posterior mean and standard error of one main effect on a grid.

        Evaluates :math:`\\hat{f}_j(g) = \\mathbf{G}\\hat{\\beta}_j` and its
        pointwise standard error :math:`\\sqrt{\\operatorname{diag}(
        \\mathbf{G}\\Sigma_{jj}\\mathbf{G}^\\top)}`, where :math:`\\mathbf{G}`
        is the component's basis evaluated on ``grid``.

        Parameters
        ----------
        feature : int or str
            Feature index, or a name present in ``feature_names_``.

        grid : array-like of shape (n_grid,), default=None
            Points at which to evaluate. When ``None``, an evenly spaced grid
            spanning the observed training range of the feature is used.

        n_grid : int, default=200
            Number of grid points when ``grid`` is ``None``.

        Returns
        -------
        grid : ndarray of shape (n_grid,)
            Evaluation points.

        mean : ndarray of shape (n_grid,)
            Posterior mean of the component.

        se : ndarray of shape (n_grid,)
            Pointwise posterior standard error. A band is
            ``mean ± z * se``.

        Notes
        -----
        These are *pointwise* conditional bands, not simultaneous bands, and
        they condition on the estimated hyperparameters and the selected
        interaction set.

        Outside the observed training range the spline basis uses clamped
        (constant) extrapolation, so the curve flattens rather than diverging.
        That is a safety property, not evidence about the unobserved region.
        """
        if isinstance(feature, str):
            j = self.feature_names_.index(feature)
        else:
            j = feature
        spec = self._spec_of_feature(j)
        blk = self.blocks_[self.specs_.index(spec)]
        if grid is None:
            if isinstance(blk, FactorBlock):
                # Discrete levels only; a dense linspace is meaningless for
                # drop-one dummies (off-level points map to the reference).
                grid = np.asarray(blk.levels_, dtype=float)
            else:
                lo, hi = self._xmin[j], self._xmax[j]
                grid = np.linspace(lo, hi, n_grid)
        else:
            grid = np.asarray(grid, dtype=float)
        G = blk.transform(grid)
        mean = G @ self.beta_[spec.sl]
        cov = self.Sigma_[spec.sl, spec.sl]
        se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", G, cov, G), 0, None))
        return grid, mean, se

    # ---------------------------------------------------------------- fit
    def fit(self, X, y):
        """Fit the additive model.

        Builds one basis block per feature, optimizes the evidence, screens
        pairwise interactions against the working residuals, then refits main
        effects and retained interactions jointly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must be dense, numeric and finite. Column names are
            used as component labels when ``X`` is a DataFrame.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is not two-dimensional, if ``X`` and ``y`` have
            incompatible lengths, or if ``feature_names`` has a length other
            than the number of columns.
        """
        # Capture sklearn metadata from original X (before coercion).
        self._set_sklearn_fit_attrs(X)
        X = self._coerce(X)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have incompatible lengths: {X.shape[0]} vs {y.shape[0]}"
            )
        self._names(X.shape[1])
        self._xmin, self._xmax = X.min(axis=0), X.max(axis=0)

        Z = self._fit_main_blocks(X)
        fit = self._fit_engine(Z, y)

        if self.interactions in ("auto", True):
            resid = self._working_residuals(Z, y, fit)
            chosen = self._screen_interactions(X, Z, resid)
            if chosen:
                Z = self._add_interaction_blocks(X, Z, chosen)
                fit = self._fit_engine(Z, y)
            else:
                self.interactions_ = []
        else:
            self.interactions_ = []

        self._Ztrain = Z
        self.beta_ = fit["beta"]
        self.Sigma_ = fit["Sigma"]
        self.evidence_ = fit["evidence"]
        self.evidence_history_ = fit["history"]
        self.edf_ = fit["edf"]
        self.n_iter_ = fit["n_iter"]
        self.at_boundary_ = self._flag_boundary_blocks()
        self._finalize(y, fit)
        return self

    #: A component using less than half an effective parameter is treated as
    #: driven to the ARD boundary rather than estimated: its precision is
    #: still growing with ``max_iter`` and is not a converged quantity.
    #:
    #: Calibrated at the *default* ``max_iter`` / ``tol``, which is where it
    #: has to work.  Over 8 seeds an exactly irrelevant orthogonal feature
    #: reached edf 0.05-0.46, while the weakest genuinely-present signal
    #: tested (coefficient 0.05 against noise 0.3) reached 0.71-0.94 -- so
    #: 0.5 separates them with margin on both sides.  An earlier value of
    #: 0.01 was calibrated against a long diagnostic run instead and missed
    #: every default-settings case it existed to catch.
    _BOUNDARY_EDF = 0.5

    def _flag_boundary_blocks(self):
        """Names of components sitting at the ARD boundary (precision -> inf).

        The evidence for a component with no signal is maximized at infinite
        precision, so the EM iteration never reaches an interior optimum --
        ``kappa`` grows roughly linearly in the iteration count while the
        evidence trace flattens. The loop then stops on the evidence
        tolerance, so the reported precision reflects ``tol`` and
        ``max_iter`` rather than the data. edf is the stable signal: it goes
        to zero regardless of where the iteration was cut off.
        """
        return sorted(
            name for name, e in self.edf_.items()
            if np.isfinite(e) and e < self._BOUNDARY_EDF
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.input_tags.sparse = False
        return tags


class SRAERegressor(RegressorMixin, _BaseSRAE):
    """Self-regularizing additive estimator for regression.

    Fits an order-two functional-ANOVA model with penalized-spline main
    effects and evidence-screened pairwise interactions,

    .. math::

        y = \\beta_0 + \\sum_j f_j(x_j)
            + \\sum_{(j,k) \\in \\mathcal{S}} f_{jk}(x_j, x_k) + \\varepsilon,
        \\qquad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2).

    Roughness precisions, null-space precisions and the residual variance are
    estimated by maximizing the log marginal likelihood (the *evidence*) with
    closed-form EM updates, so no cross-validated penalty grid is required.
    Interactions are discovered automatically by evidence-gain screening.

    Read more in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    n_knots : int, default=10
        Number of interior knots for each spline block, placed at empirical
        quantiles. Under ``feature_types='auto'``, features with at most
        ``max_linear_card`` distinct values receive a single standardized
        linear column instead. Larger values give a finer basis; the effective
        smoothness is still set by the evidence, so this mainly bounds
        resolution rather than complexity.

    max_linear_card : int, default=5
        Auto-dispatch only: features with at most this many distinct values
        receive a single standardized :class:`~srae.blocks.LinearBlock` column
        instead of a spline. That column is **linear in the numeric codes**, so
        nominal labels and non-monotone low-cardinality effects (e.g. a
        U-shape) are misspecified and tend to fit flat. Prefer
        ``feature_types='factor'`` (or a per-column dict entry) for those
        columns; lower this threshold or set ``0`` to force splines under auto.

    feature_types : None, str, sequence of str, or dict, default=None
        Per-feature block choice. ``None`` (and the string ``'auto'``) uses the
        ``max_linear_card`` heuristic for every column. A single string applies
        to all columns; a sequence must have one entry per column; a dict maps
        feature names or integer indices to types (unlisted columns stay
        ``'auto'``).

        Canonical types (aliases in parentheses):

        * ``'auto'`` — cardinality heuristic above.
        * ``'linear'`` (``'ordinal'``) — :class:`~srae.blocks.LinearBlock`.
        * ``'spline'`` (``'continuous'``, ``'numeric'``) —
          :class:`~srae.blocks.SplineBlock`.
        * ``'factor'`` (``'categorical'``, ``'nominal'``, ``'cat'``) —
          :class:`~srae.blocks.FactorBlock` (drop-one dummies + shared ARD).

        Resolved types are stored on ``feature_types_`` after ``fit``.

    interactions : {'auto', True, False}, default='auto'
        Whether to screen for pairwise interactions. ``False`` fits a purely
        additive model and leaves ``interactions_`` empty.

    max_interactions : int, default=8
        Maximum number of interaction surfaces retained after thresholding.

    max_screen_pairs : int, default=40
        Number of candidate pairs scored by full evidence. When
        ``p * (p - 1) / 2`` exceeds this, a product-correlation pre-filter
        selects which pairs to score. Set to the full pair count to disable
        the pre-filter.

    interaction_gain_threshold : float, default=4.0
        Minimum evidence gain, in nats, for a pair to be retained. This is a
        structural setting, not a calibrated false-discovery control. The
        default is also calibrated against a specific tensor basis convention
        -- a ridge on tensor B-spline coefficients is not invariant to how the
        marginals are parametrized -- so it does not transfer to a differently
        parametrized tensor basis. See ``docs/user_guide/interactions.rst``.

    max_iter : int, default=200
        Maximum number of EM iterations.

    tol : float, default=1e-5
        Relative tolerance on the evidence for the EM stopping rule.

    feature_names : list of str, default=None
        Component labels. When ``None``, names are taken from the columns of a
        DataFrame ``X``, or generated as ``x0 … xp``. This parameter is never
        modified by ``fit``.

    verbose : bool, default=False
        Print the evidence trace during fitting.

    Attributes
    ----------
    beta_ : ndarray of shape (n_basis,)
        Posterior mean of the stacked basis coefficients.

    Sigma_ : ndarray of shape (n_basis, n_basis)
        Posterior covariance of ``beta_``.

    sigma2_ : float
        Estimated residual variance :math:`\\sigma^2`.

    evidence_ : float
        Final log marginal likelihood.

    evidence_history_ : ndarray of shape (n_iter\\_ + 1,)
        Evidence at each EM iteration; useful as a convergence diagnostic.

    edf_ : dict of str to float
        Effective degrees of freedom per component, :math:`\\sum_i
        (1 - a_i \\Sigma_{ii})` over the block.

    at_boundary_ : list of str
        Components driven to the ARD boundary (edf below
        ``_BOUNDARY_EDF``, default 0.5 at the shipped settings). Their
        evidence is maximized at infinite precision, so the reported
        ``lam`` / ``kap`` keep growing with ``max_iter`` and are not
        converged estimates -- read them as "pruned to zero", and use
        ``edf_`` (which is stable) to judge contribution.

    n_iter_ : int
        Number of EM iterations actually run.

    interactions_ : list of dict
        Retained interactions, each with keys ``'pair'``, ``'name'`` and
        ``'screen_gain'``.

    blocks_ : list
        Fitted basis blocks, aligned with ``specs_``.

    specs_ : list of BlockSpec
        Column ranges, penalty eigenvalues and hyperparameters per block.

    feature_names_ : list of str
        Component labels resolved at fit time.

    feature_types_ : list of str
        Canonical per-column types resolved at fit time
        (``'auto'``, ``'linear'``, ``'spline'``, or ``'factor'``). Note that
        ``'auto'`` records the *request*, not the block that was built; inspect
        ``blocks_`` for the realized type.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (n_features_in\\_,)
        Column names seen during ``fit``. Defined only when ``X`` has them.

    See Also
    --------
    SRAEClassifier : Classification counterpart.
    SRAERegressorPooled : Adds the pooled anti-overfitting stack.
    SRAERegressorSI : Integrates over global hyperparameter scales.

    Notes
    -----
    ``X`` must be dense, numeric and finite; missing values, string
    categoricals and sparse matrices are the caller's responsibility. Numeric
    codes for nominal levels can be modelled in-estimator with
    ``feature_types='factor'``. Feature scaling is not required, since knots
    are quantile-based and every block is internally standardized.

    Predictive intervals include intercept sampling variance
    (:math:`\\sigma^2 / n`) but still condition on the estimated
    hyperparameters, the selected interaction set and the fixed basis. They
    do not propagate interaction-selection uncertainty; see
    :meth:`predict_interval` for coverage behaviour at small :math:`n`.

    References
    ----------
    .. [1] P. H. C. Eilers and B. D. Marx, "Flexible smoothing with B-splines
           and penalties", *Statistical Science*, 11(2):89-121, 1996.
    .. [2] D. J. C. MacKay, "Bayesian interpolation", *Neural Computation*,
           4(3):415-447, 1992.
    .. [3] S. N. Wood, *Generalized Additive Models: An Introduction with R*,
           2nd ed., Chapman and Hall/CRC, 2017.

    Examples
    --------
    >>> import numpy as np
    >>> from srae import SRAERegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = np.sin(1.5 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.normal(size=200)
    >>> model = SRAERegressor(interactions=False).fit(X, y)
    >>> bool(model.score(X, y) > 0.8)
    True
    >>> grid, mean, se = model.shape_function(0)
    """

    _needs_intercept = False

    def _fit_engine(self, Z, y):
        self._ymean = float(np.mean(y))
        self._n_train = int(len(y))
        return fit_gaussian_eb(Z, y - self._ymean, self.specs_,
                               max_iter=self.max_iter, tol=self.tol,
                               verbose=self.verbose)

    def _working_residuals(self, Z, y, fit):
        return (y - self._ymean) - Z @ fit["beta"]

    def _finalize(self, y, fit):
        self.sigma2_ = fit["sigma2"]

    def _intercept_sampling_variance(self, sigma2=None):
        """Sampling variance of the response-mean intercept, :math:`\\sigma^2/n`.

        The regressor centers ``y`` at ``ybar`` and adds it back at predict
        time, so ``ybar`` is estimated. Its contribution to predictive
        variance is :math:`\\sigma^2 / n_{\\mathrm{train}}`.
        """
        n = getattr(self, "_n_train", None)
        if n is None or int(n) < 1:
            raise RuntimeError(
                "fitted regressor is missing _n_train; every regression "
                "_fit_engine must record the training sample size so the "
                "intercept sampling variance sigma^2/n can be included in "
                "predictive intervals"
            )
        if sigma2 is None:
            sigma2 = self.sigma2_
        return float(sigma2) / int(n)

    def predict(self, X, return_std=False):
        """Predict the response, optionally with a predictive standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        return_std : bool, default=False
            If ``True``, also return the standard deviation of the posterior
            predictive distribution.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Posterior mean prediction.

        y_std : ndarray of shape (n_samples,)
            Predictive standard deviation, returned only when ``return_std``
            is ``True``. Combines parameter uncertainty, observation noise,
            and intercept sampling variance:
            :math:`\\mathbf{z}^\\top \\Sigma \\mathbf{z} + \\sigma^2
            + \\sigma^2 / n_{\\mathrm{train}}`.
        """
        if not hasattr(self, "beta_"):
            raise RuntimeError("This SRAERegressor instance is not fitted yet. "
                               "Call 'fit' with appropriate arguments before using this estimator.")
        Z = self._design(self._coerce(X))
        mean = self._ymean + Z @ self.beta_
        if not return_std:
            return mean
        var_f = np.einsum("ij,jk,ik->i", Z, self.Sigma_, Z)
        # The intercept is estimated (the response is centered at ybar), so its
        # sampling variance sigma^2 / n_train belongs in the predictive
        # variance.  It is small -- 1/n of the noise term -- but omitting it
        # made the intercept the one fitted quantity treated as known.
        std = np.sqrt(
            np.clip(var_f, 0, None)
            + self.sigma2_
            + self._intercept_sampling_variance()
        )
        return mean, std

    def predict_interval(self, X, level=0.9):
        """Equal-tailed Gaussian predictive interval.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        level : float, default=0.9
            Nominal coverage, strictly between 0 and 1.

        Returns
        -------
        lower : ndarray of shape (n_samples,)
            Lower interval endpoint.

        upper : ndarray of shape (n_samples,)
            Upper interval endpoint.

        Notes
        -----
        Conditions on the estimated hyperparameters, the selected interaction
        set and the fixed basis; selection uncertainty is not propagated.

        **Coverage is optimistic at small n, and the shortfall is an
        n-effect.** On a well-specified Gaussian design (3 features, smooth
        additive truth, :math:`\\sigma^2 = 0.25`, 40-60 replicates), nominal
        90% intervals covered:

        ===========  ==========  ==============================
        ``n_train``  coverage    :math:`\\hat{\\sigma}^2` (true 0.25)
        ===========  ==========  ==============================
        60           85.1%       0.225
        100          87.4%       0.237
        200          89.3%       0.250
        400          89.8%       0.252
        ===========  ==========  ==============================

        The cause is *not* a missing degrees-of-freedom correction:
        :math:`\\hat{\\sigma}^2` already equals
        :math:`\\mathrm{RSS}/(n - \\mathrm{edf})` at the EM fixed point, since
        :math:`\\operatorname{tr}(\\Sigma \\mathbf{Z}^\\top\\mathbf{Z}) =
        \\sigma^2 \\, \\mathrm{edf}`.  What is missing is uncertainty in the
        *smoothing parameters* themselves, which are estimated from the same
        data and then conditioned upon -- a known shortfall for empirical-Bayes
        GAM intervals (see Wood, *Generalized Additive Models*, 2nd ed., on
        smoothing-parameter uncertainty).  Interaction-selection uncertainty is
        a separate concern and contributed nothing measurable in this
        experiment.

        Integrating the hyperparameter scale recovers most of the gap: at
        :math:`n = 100` :class:`~srae.SRAERegressorSI` reached 89.3% against
        87.4% here, and 90.5% versus 89.3% at :math:`n = 200`.  Prefer the
        scale-integrated variants when interval calibration matters at small
        :math:`n`.
        """
        from scipy.stats import norm

        mean, std = self.predict(X, return_std=True)
        z = norm.ppf(0.5 + level / 2)
        return mean - z * std, mean + z * std

    def score(self, X, y):
        """Coefficient of determination :math:`R^2` of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        score : float
            :math:`R^2`, with 1.0 the best possible value. May be negative
            for a model worse than the mean predictor.
        """
        y = np.asarray(y, float).ravel()
        pred = self.predict(X)
        ss = float(np.sum((y - pred) ** 2))
        return 1.0 - ss / float(np.sum((y - y.mean()) ** 2))


class SRAEClassifier(ClassifierMixin, _BaseSRAE):  # noqa: D101 - see class docstring
    """Self-regularizing additive estimator for classification.

    Binary problems use a Bernoulli-logit likelihood with Laplace-approximate
    evidence optimization; the additive predictor is the same order-two
    functional-ANOVA decomposition used for regression,

    .. math::

        \\Pr(y = 1 \\mid \\mathbf{x}) = \\operatorname{sigmoid}\\!\\left(
            \\beta_0 + \\sum_j f_j(x_j)
            + \\sum_{(j,k) \\in \\mathcal{S}} f_{jk}(x_j, x_k)\\right).

    Multiclass problems are handled by a one-vs-rest ensemble of binary SRAEs
    whose log-odds are coupled through a softmax. Each per-class model remains a
    fully transparent additive model, exposed through ``estimators_``.

    For binary fits, predicted probabilities are *moderated*: the posterior
    variance of the linear predictor enters the link, so predictions shrink
    toward 0.5 where the model is least certain. The default multiclass path
    instead applies a softmax to the independently fitted one-vs-rest
    log-odds; it does not apply binary moderation to each head.

    Read more in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    n_knots : int, default=10
        Number of interior knots for each spline block, placed at empirical
        quantiles. Under ``feature_types='auto'``, features with at most
        ``max_linear_card`` distinct values receive a linear column instead.

    max_linear_card : int, default=5
        Auto-dispatch only: features with at most this many distinct values
        receive a :class:`~srae.blocks.LinearBlock`. That column is linear in
        the numeric codes; use ``feature_types='factor'`` for nominal or
        non-monotone low-cardinality effects. ``0`` disables the auto rule.

    feature_types : None, str, sequence of str, or dict, default=None
        Per-feature block choice (``'auto'``, ``'linear'``, ``'spline'``,
        ``'factor'`` or aliases). See :class:`SRAERegressor` for the full
        specification. Resolved types are stored on ``feature_types_``.

    interactions : {'auto', True, False}, default='auto'
        Whether to screen for pairwise interactions. ``False`` fits a purely
        additive model.

    max_interactions : int, default=8
        Maximum number of interaction surfaces retained after thresholding.

    max_screen_pairs : int, default=40
        Number of candidate pairs scored by full evidence before the
        product-correlation pre-filter engages.

    interaction_gain_threshold : float, default=4.0
        Minimum evidence gain, in nats, for a pair to be retained. Calibrated
        against SRAE's tensor basis convention and not transferable to a
        differently parametrized one; see ``docs/user_guide/interactions.rst``.

    max_iter : int, default=200
        Maximum number of outer evidence iterations. Internally capped at 100
        for the logistic engine.

    tol : float, default=1e-5
        Relative tolerance on the evidence for the stopping rule.

    feature_names : list of str, default=None
        Component labels. Never modified by ``fit``.

    verbose : bool, default=False
        Print the evidence trace during fitting.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    estimators_ : list of SRAEClassifier or None
        One-vs-rest sub-models for multiclass problems; ``None`` for binary.

    joint_ : dict or None
        The joint multinomial Laplace fit for multiclass problems: ``beta``
        of shape ``(K-1, n_columns)`` in the sum-to-zero contrast basis,
        ``Sigma``, ``contrasts``, ``edf``, ``evidence``, ``history``,
        ``n_iter``. Absent for binary fits, and ``None`` when the joint refit
        was declined (see ``_MAX_JOINT_DIM``) or failed.

    multiclass_link : {'joint', 'softmax', 'normalized_ovr'}, class attribute
        How class probabilities are produced. ``'joint'`` (default since
        0.0.10) predicts from the joint multinomial posterior in ``joint_``,
        moderating toward ``1/K``. ``'softmax'`` (the 0.0.6-0.0.9 default)
        softmaxes the stacked one-vs-rest log-odds with no moderation;
        ``'normalized_ovr'`` (to 0.0.5) moderates each head and divides by the
        row sum. Both legacy routes measured worse than ``'joint'`` on
        held-out log-loss; ``'softmax'`` measured worst of the three. Set on
        the instance to reproduce older results; not a constructor parameter
        and not returned by ``get_params``.

    beta_ : ndarray of shape (n_basis,)
        Posterior mode of the stacked coefficients. Binary models only.

    Sigma_ : ndarray of shape (n_basis, n_basis)
        Laplace posterior covariance. Binary models only.

    evidence_ : float
        Final approximate log marginal likelihood. For multiclass, the sum
        over sub-models.

    evidence_history_ : ndarray
        Evidence at each iteration. May be non-monotone under the Laplace
        approximation. Binary models only; for multiclass inspect each object
        in ``estimators_``.

    edf_ : dict of str to float
        Effective degrees of freedom per component. Binary models only; for
        multiclass inspect each object in ``estimators_``.

    at_boundary_ : list of str
        Components driven to the ARD boundary (edf below
        ``_BOUNDARY_EDF``, default 0.5 at the shipped settings). Their
        evidence is maximized at infinite precision, so the reported
        ``lam`` / ``kap`` keep growing with ``max_iter`` and are not
        converged estimates -- read them as "pruned to zero", and use
        ``edf_`` (which is stable) to judge contribution. Binary models only;
        for multiclass inspect each object in ``estimators_``.

    n_iter_ : int
        Number of iterations actually run. Binary models only; for multiclass
        inspect each object in ``estimators_``.

    interactions_ : list of dict
        Retained interactions. For multiclass, entries carry a ``'cls'`` key.

    feature_names_ : list of str
        Component labels resolved at fit time.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (n_features_in\\_,)
        Column names seen during ``fit``. Defined only when ``X`` has them.

    See Also
    --------
    SRAERegressor : Regression counterpart.
    SRAEClassifierPooled : Adds the pooled anti-overfitting stack.
    SRAEClassifierSI : Integrates over global hyperparameter scales.

    Notes
    -----
    The Laplace approximation makes this path approximate: the monotone-ascent
    guarantee of the Gaussian EM argument does not carry over.

    Multiclass fits remain a one-vs-rest ensemble: the heads are estimated
    under K independent Bernoulli likelihoods, and ``evidence_`` is their sum,
    so interaction screening is still uncoupled across classes. Only the
    reported probabilities are coupled, through a softmax over the stacked
    one-vs-rest log-odds (see ``multiclass_link``). A joint multinomial Laplace
    approximation would couple the estimation as well.

    Interaction screening has markedly less power under a Bernoulli likelihood
    than under a Gaussian one. An empty ``interactions_`` on a small
    classification problem indicates lack of power, not absence of structure.

    References
    ----------
    .. [1] D. J. C. MacKay, "The evidence framework applied to classification
           networks", *Neural Computation*, 4(5):720-736, 1992.
    .. [2] C. M. Bishop, *Pattern Recognition and Machine Learning*,
           chapters 3-4, Springer, 2006.

    Examples
    --------
    >>> import numpy as np
    >>> from srae import SRAEClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> f = np.sin(1.5 * X[:, 0]) + X[:, 1] ** 2
    >>> y = (rng.uniform(size=200) < 1 / (1 + np.exp(-(f - f.mean())))).astype(int)
    >>> clf = SRAEClassifier(interactions=False).fit(X, y)
    >>> clf.predict_proba(X).shape
    (200, 2)
    """

    _needs_intercept = True

    #: How one-vs-rest heads become class probabilities; see _multiclass_proba.
    #: 'normalized_ovr' restores the pre-2026-07 row-normalisation behaviour.
    multiclass_link = "joint"

    def _fit_engine(self, Z, y):
        return fit_logistic_eb(Z, y, self.specs_,
                               max_iter=min(self.max_iter, 100), tol=self.tol,
                               verbose=self.verbose)

    def _working_residuals(self, Z, y, fit):
        # Gradient-space pseudo-residuals for interaction screening.
        return y - _sigmoid(Z @ fit["beta"])

    def _finalize(self, y, fit):
        pass

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        """Fit the classifier.

        Binary targets are fitted directly. Targets with more than two classes
        build a one-vs-rest ensemble, one independent binary SRAE per class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must be dense, numeric and finite.

        y : array-like of shape (n_samples,)
            Class labels. Any hashable label type is supported; the sorted
            unique values become ``classes_``.

        Returns
        -------
        self : object
            The fitted estimator.

        Raises
        ------
        ValueError
            If fewer than two classes are present, or if ``X`` and ``y`` have
            incompatible lengths.
        """
        self._set_sklearn_fit_attrs(X)
        Xa = self._coerce(X)
        y = np.asarray(y).ravel()
        if Xa.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have incompatible lengths: {Xa.shape[0]} vs {y.shape[0]}"
            )
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least 2 classes")
        if len(self.classes_) == 2:
            self.estimators_ = None
            y01 = (y == self.classes_[1]).astype(float)
            # Pass the original X: _BaseSRAE.fit re-runs _set_sklearn_fit_attrs
            # (idempotent), so column names survive instead of degrading to x0..xp.
            _BaseSRAE.fit(self, X, y01)
            return self
        # one-vs-rest ensemble for multiclass
        self.estimators_ = []
        for c in self.classes_:
            est = SRAEClassifier(
                n_knots=self.n_knots, max_linear_card=self.max_linear_card,
                feature_types=self.feature_types,
                interactions=self.interactions,
                max_interactions=self.max_interactions,
                max_screen_pairs=self.max_screen_pairs,
                interaction_gain_threshold=self.interaction_gain_threshold,
                max_iter=self.max_iter, tol=self.tol,
                feature_names=list(self.feature_names_), verbose=self.verbose,
            )
            est.fit(Xa, (y == c).astype(float))
            self.estimators_.append(est)
        self.interactions_ = [
            dict(cls=c, **info)
            for c, est in zip(self.classes_, self.estimators_)
            for info in est.interactions_
        ]
        self._fit_joint_multinomial(Xa, y)
        return self

    def _fit_joint_multinomial(self, X, y):
        """Refit the selected structure as one joint multinomial model.

        The one-vs-rest pass above discovers structure -- which blocks, which
        interaction pairs -- and that part stays uncoupled, as documented. What
        it cannot give is a joint posterior: independent binary fits leave the
        cross-class blocks of the Hessian at zero, so there is no coherent
        covariance between class surfaces and no neutral point to moderate
        toward. This refits the union of the discovered structure under one
        softmax likelihood, which supplies both.

        Falls back to the stacked one-vs-rest route if the joint fit fails
        numerically, so a hard problem degrades rather than raising.
        """
        Z = self._fit_main_blocks(X)
        pairs, seen = [], set()
        for info in self.interactions_:
            if info["pair"] not in seen:
                seen.add(info["pair"])
                pairs.append(info["pair"])
        chosen = []
        for pair in pairs:
            tb = TensorBlock(pair, levels=self._pair_levels(pair))
            try:
                tb.fit(X[:, pair[0]], X[:, pair[1]],
                       self._main_design_for_pair(Z, pair))
            except ValueError:
                continue
            gain = next(i["screen_gain"] for i in self.interactions_
                        if i["pair"] == pair)
            chosen.append((gain, pair, tb))
        joint_interactions = self.interactions_
        if chosen:
            Z = self._add_interaction_blocks(X, Z, chosen)
        self.interactions_ = joint_interactions

        # Warm-start each shared precision from the one-vs-rest children. They
        # have already located the right order of magnitude per component, and
        # the joint EM step is far more expensive than a binary one -- its
        # Hessian is (K-1)p square and couples every class -- so starting from
        # 1.0 would spend most of the budget re-deriving what is already known.
        # Geometric mean, because these are precisions spanning many decades.
        by_name = {}
        for est in self.estimators_:
            for sp in getattr(est, "specs_", []):
                by_name.setdefault(sp.name, []).append((sp.lam, sp.kap))
        for spec in self.specs_:
            vals = by_name.get(spec.name)
            if not vals or isinstance(spec, _InterceptSpec):
                continue
            lams = np.array([v[0] for v in vals], float)
            kaps = np.array([v[1] for v in vals], float)
            spec.lam = float(np.exp(np.mean(np.log(np.clip(lams, 1e-300, None)))))
            spec.kap = float(np.exp(np.mean(np.log(np.clip(kaps, 1e-300, None)))))

        # The joint Hessian is ((K-1) * n_columns) square and is factorized
        # inside every EM step, so cost grows as its cube and memory as its
        # square. Past a few thousand it stops being a refinement and starts
        # being the whole fit -- and can exhaust memory outright. Decline
        # rather than degrade silently: the caller keeps a working model on the
        # better-calibrated legacy link and is told why.
        dim = (len(self.classes_) - 1) * Z.shape[1]
        if dim > _MAX_JOINT_DIM:
            warnings.warn(
                f"skipping the joint multinomial refit: it would need a "
                f"{dim}x{dim} Hessian, above the {_MAX_JOINT_DIM} limit "
                f"({len(self.classes_)} classes x {Z.shape[1]} columns). "
                f"Falling back to the one-vs-rest 'normalized_ovr' link; "
                f"probabilities will not be moderated toward 1/K. Reduce "
                f"n_knots, max_interactions, or the number of features to "
                f"enable the joint fit.",
                RuntimeWarning, stacklevel=3,
            )
            self._joint_ = None
            self.evidence_ = float(sum(e.evidence_ for e in self.estimators_))
            return

        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        try:
            fit = fit_multinomial_eb(Z, Y, self.specs_,
                                     max_iter=min(self.max_iter, 100),
                                     tol=self.tol, verbose=self.verbose)
        except (np.linalg.LinAlgError, ValueError) as exc:
            warnings.warn(
                f"joint multinomial fit failed ({exc}); falling back to the "
                f"stacked one-vs-rest link. Probabilities will not be "
                f"moderated toward 1/K.",
                RuntimeWarning, stacklevel=2,
            )
            self._joint_ = None
            self.evidence_ = float(sum(e.evidence_ for e in self.estimators_))
            return

        # Deliberately does *not* overwrite beta_ / Sigma_ / edf_ / n_iter_ on
        # the parent. Those are documented as binary-only, and the one-vs-rest
        # children stay the per-class structural view that summary(),
        # shape_function() and the plotting helpers report: each child carries
        # an absolute surface per class, whereas the joint model is a contrast
        # parametrization against a reference class and has no coefficients for
        # it at all. Mixing the two in one report would be worse than keeping
        # them separate. ``evidence_`` does become the joint value, since a
        # single coherent marginal likelihood strictly beats a sum of
        # independent ones.
        self._joint_ = fit
        self.joint_ = dict(
            beta=fit["beta"], Sigma=fit["Sigma"], edf=fit["edf"],
            evidence=fit["evidence"], history=fit["history"],
            n_iter=fit["n_iter"], contrasts=fit["contrasts"],
        )
        self.evidence_ = fit["evidence"]

    def _joint_logits_and_variance(self, X):
        """Sum-to-zero class logits and the mean pairwise contrast variance.

        Returns ``(Eta, vbar)`` where ``Eta`` is ``(n, K)`` and ``vbar``
        averages ``Var(eta_k - eta_l)`` over unordered class pairs.  Averaging
        *contrast* variances rather than per-class ones keeps the moderation
        independent of any labelling convention, and reduces to the binary
        posterior variance at ``K = 2``.
        """
        Zt = np.asarray(self._design(X), float)
        G = self._joint_["beta"]                 # (K-1, p) contrast coefs
        C = self._joint_["contrasts"]            # (K, K-1)
        n_heads, p = G.shape
        Sigma = self._joint_["Sigma"]
        Eta = Zt @ G.T @ C.T                     # (n, K)

        # g[a, b] = diag(Z Sigma_ab Z') in contrast coordinates.
        n = Zt.shape[0]
        g = np.empty((n_heads, n_heads, n))
        for a in range(n_heads):
            for b in range(a, n_heads):
                blk = Sigma[a * p:(a + 1) * p, b * p:(b + 1) * p]
                gab = np.einsum("ij,jk,ik->i", Zt, blk, Zt)
                g[a, b] = gab
                g[b, a] = gab
        # Var(eta_k - eta_l) = (C_k - C_l)' g (C_k - C_l)
        K = n_heads + 1
        total = np.zeros(n)
        for k in range(K):
            for l in range(k + 1, K):
                dvec = C[k] - C[l]
                total += np.einsum("a,ab...,b->...", dvec, g, dvec)
        return Eta, total / max(K * (K - 1) / 2, 1)

    @property
    def _is_multiclass(self):
        return getattr(self, "estimators_", None) is not None

    def decision_function(self, X):
        """Value of the additive predictor (the log-odds).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray
            Shape ``(n_samples,)`` for binary problems, or
            ``(n_samples, n_classes)`` of per-class one-vs-rest scores for
            multiclass problems.
        """
        if not hasattr(self, "classes_"):
            raise RuntimeError("This SRAEClassifier instance is not fitted yet. "
                               "Call 'fit' with appropriate arguments before using this estimator.")
        if self._is_multiclass:
            return np.column_stack([e.decision_function(X)
                                    for e in self.estimators_])
        Z = self._design(self._coerce(X))
        return Z @ self.beta_

    def predict_proba(self, X):
        """Class probabilities.

        For a binary fit, rather than plugging the posterior mode into the
        link, the Gaussian posterior of the linear predictor is integrated
        against it using the probit approximation

        .. math::

            \\Pr(y = 1 \\mid \\mathbf{z}) \\approx
            \\operatorname{sigmoid}\\!\\left(
                \\frac{\\mu}{\\sqrt{1 + \\pi \\nu / 8}}\\right),
            \\qquad
            \\mu = \\mathbf{z}^\\top \\hat{\\beta}, \\;
            \\nu = \\mathbf{z}^\\top \\Sigma \\mathbf{z},

        which shrinks predictions toward 0.5 where the model is uncertain.
        For a multiclass fit, the default route predicts from the joint
        multinomial posterior and shrinks toward ``1/K`` instead. See
        ``multiclass_link`` and :meth:`_multiclass_proba`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities, rows summing to one. For multiclass, the
            stacked one-vs-rest log-odds are passed through a softmax; see
            ``multiclass_link`` and :meth:`_multiclass_proba`.
        """
        if not hasattr(self, "classes_"):
            raise RuntimeError("This SRAEClassifier instance is not fitted yet. "
                               "Call 'fit' with appropriate arguments before using this estimator.")
        if self._is_multiclass:
            return self._multiclass_proba(X)
        Z = self._design(self._coerce(X))
        mu = Z @ self.beta_
        var = np.clip(np.einsum("ij,jk,ik->i", Z, self.Sigma_, Z), 0, None)
        p1 = _sigmoid(mu / np.sqrt(1.0 + np.pi * var / 8.0))
        return np.column_stack([1 - p1, p1])

    def _multiclass_logits(self, X):
        """Stacked one-vs-rest linear predictors, shape ``(n_samples, n_classes)``."""
        return np.column_stack([e.decision_function(X) for e in self.estimators_])

    def _multiclass_proba(self, X):
        """Class probabilities from the joint multinomial posterior.

        ``multiclass_link="joint"`` (the default since 0.0.10) uses the
        Laplace posterior of the joint model fitted by
        :meth:`_fit_joint_multinomial`, moderating the logits toward the
        ``K``-class neutral point ``1/K``:

        .. code-block:: text

            p = softmax(eta / sqrt(1 + pi * vbar / 8))

        with ``vbar`` the mean variance of the pairwise logit contrasts.  A
        *common* factor per row -- rather than one per class -- is what keeps
        this independent of which class is the reference, since adding a
        constant to every logit leaves a softmax unchanged.  At ``K = 2`` it
        reduces exactly to the binary moderated probability, and the joint
        engine itself reduces to the logistic one.

        Two legacy routes remain, for reproducing published results:

        ``"softmax"`` (the 0.0.6-0.0.9 default) applies a softmax to the
        stacked one-vs-rest log-odds.  It guarantees coherent rows but drops
        moderation entirely, and measured on held-out synthetic multiclass data
        it is *worse* calibrated than the route it replaced -- 2-4x the
        expected calibration error, and higher log-loss, on 5/5 seeds at every
        ``K`` and ``n`` tried.  Coherent row sums are not calibrated
        probabilities.

        ``"normalized_ovr"`` (to 0.0.5) moderates each head, then divides by
        the row sum.  Its row sums genuinely are incoherent -- measured between
        0.37 and 1.83 here, and as wide as 0.39 to 3.56 on a 10-class problem.
        Note, though, that dividing by the row sum maps a uniform shrinkage
        toward 0.5 onto a shrinkage toward ``1/K``, so the "wrong neutral
        point" objection raised against it in 0.0.6 was overstated; what
        remains is the *heterogeneity* of the per-head shrinkage.

        Neither legacy route yields a joint posterior: independent binary fits
        leave the cross-class Hessian blocks at zero, so no coherent
        covariance between class surfaces exists to moderate with.

        Notes
        -----
        ``multiclass_link`` is a class attribute rather than a constructor
        parameter, so it is deliberately not part of ``get_params`` and does not
        survive :func:`sklearn.base.clone`.  It exists to reproduce results
        published before these changes, not as a quantity to tune.
        """
        link = getattr(self, "multiclass_link", "joint")
        if link == "joint" and getattr(self, "_joint_", None) is None:
            # No joint fit available: the pooled and scale-integrated variants
            # override the multiclass fit with their own machinery (edf budget,
            # MH sampling over scales) and have no joint analogue yet, and a
            # joint fit can fail numerically. Fall back to ``normalized_ovr``
            # rather than ``softmax``: incoherent row sums are the lesser
            # defect, since softmax measured worse on both log-loss and ECE.
            link = "normalized_ovr"
        if link == "joint":
            Eta, vbar = self._joint_logits_and_variance(X)
            Eta = Eta / np.sqrt(1.0 + (np.pi / 8.0) * vbar)[:, None]
            Eta = Eta - Eta.max(axis=1, keepdims=True)      # overflow-safe
            P = np.exp(Eta)
            return P / P.sum(axis=1, keepdims=True)
        if link == "normalized_ovr":
            P = np.column_stack([e.predict_proba(X)[:, 1] for e in self.estimators_])
            return P / P.sum(axis=1, keepdims=True)
        eta = np.asarray(self._multiclass_logits(X), float)
        eta = eta - eta.max(axis=1, keepdims=True)   # overflow-safe
        P = np.exp(eta)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels, drawn from ``classes_``, taken as the argmax of
            :meth:`predict_proba`.
        """
        P = self.predict_proba(X)
        return self.classes_[np.argmax(P, axis=1)]

    def score(self, X, y):
        """Mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Fraction of correctly classified samples.
        """
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def summary(self):
        """Per-component report, with one section per class if multiclass.

        Returns
        -------
        summary : pandas.DataFrame
            As :meth:`SRAERegressor.summary`, with an additional leading
            ``class`` column for multiclass models.
        """
        if not self._is_multiclass:
            return super().summary()
        import pandas as pd

        frames = []
        for c, est in zip(self.classes_, self.estimators_):
            df = est.summary()
            df.insert(0, "class", c)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def shape_function(self, feature, cls=None, **kw):
        """Posterior mean and standard error of one main effect.

        Parameters
        ----------
        feature : int or str
            Feature index, or a name present in ``feature_names_``.

        cls : object, default=None
            Class label selecting which one-vs-rest sub-model to query.
            Required for multiclass models, ignored for binary ones.

        **kw : dict
            Passed through to :meth:`SRAERegressor.shape_function`
            (``grid``, ``n_grid``).

        Returns
        -------
        grid : ndarray of shape (n_grid,)
            Points at which the component was evaluated.

        mean : ndarray of shape (n_grid,)
            Posterior mean of the component on the log-odds scale.

        se : ndarray of shape (n_grid,)
            Pointwise posterior standard error.

        Raises
        ------
        ValueError
            If the model is multiclass and ``cls`` is not given.
        """
        if self._is_multiclass:
            if cls is None:
                raise ValueError("multiclass model: pass cls=<class label>")
            k = list(self.classes_).index(cls)
            return self.estimators_[k].shape_function(feature, **kw)
        return super().shape_function(feature, **kw)
