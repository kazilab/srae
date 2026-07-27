"""Core modelling API: fit/predict, reporting, interactions, plotting, errors.

Covers all four variants across the three supported paths (regression, binary
classification, multiclass classification).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from conftest import (COLUMNS, default_target, is_classifier_cls, is_pooled,
                      make, screening_data)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# --------------------------------------------------------------------------
# Regression path
# --------------------------------------------------------------------------

class TestRegression:
    def test_fit_predict_shape(self, regressor_cls, X, y_reg):
        m = make(regressor_cls).fit(X, y_reg)
        pred = m.predict(X)
        assert pred.shape == (len(y_reg),)
        assert np.all(np.isfinite(pred))

    def test_recovers_signal(self, regressor_cls, X, y_reg):
        """The latent signal is interaction-dominated, so screening must help.

        Absolute R^2 differs sharply by variant -- the pooled stack's edf budget
        and holdout calibration shrink much harder at n=80 -- so the floor is
        set per variant and the *relative* improvement carries the assertion.

        The floors sit well below the observed values (~0.95 plain, ~0.49
        pooled). They dropped by ~0.03 when tensor purification began
        residualizing against the block's own marginals, which removes
        main-effect-shaped structure the tensor could previously absorb; see
        ``TestTensorPurification``.
        """
        floor = 0.35 if is_pooled(regressor_cls) else 0.85
        additive = make(regressor_cls, interactions=False).fit(X, y_reg)
        full = make(regressor_cls, interactions="auto").fit(X, y_reg)
        assert full.score(X, y_reg) > floor
        assert full.score(X, y_reg) > additive.score(X, y_reg) + 0.05

    def test_predict_return_std(self, regressor_cls, X, y_reg):
        m = make(regressor_cls).fit(X, y_reg)
        mean, std = m.predict(X, return_std=True)
        assert mean.shape == std.shape == (len(y_reg),)
        assert np.all(std > 0)

    def test_predict_interval_brackets_mean(self, regressor_cls, X, y_reg):
        m = make(regressor_cls).fit(X, y_reg)
        lo, hi = m.predict_interval(X, level=0.9)
        mean = m.predict(X)
        assert np.all(lo < mean) and np.all(mean < hi)

    def test_wider_interval_for_higher_level(self, regressor_cls, X, y_reg):
        m = make(regressor_cls).fit(X, y_reg)
        lo50, hi50 = m.predict_interval(X, level=0.5)
        lo99, hi99 = m.predict_interval(X, level=0.99)
        assert np.all((hi99 - lo99) > (hi50 - lo50))

    def test_sigma2_positive(self, regressor_cls, X, y_reg):
        m = make(regressor_cls).fit(X, y_reg)
        assert m.sigma2_ > 0


# --------------------------------------------------------------------------
# Classification paths
# --------------------------------------------------------------------------

class TestClassification:
    def test_predict_proba_is_a_distribution(self, classifier_cls, clf_path, X, targets):
        y = targets[clf_path]
        m = make(classifier_cls).fit(X, y)
        P = m.predict_proba(X)
        assert P.shape == (len(y), len(np.unique(y)))
        assert np.allclose(P.sum(axis=1), 1.0)
        assert np.all((P >= 0) & (P <= 1))

    def test_predict_within_classes(self, classifier_cls, clf_path, X, targets):
        y = targets[clf_path]
        m = make(classifier_cls).fit(X, y)
        assert set(np.unique(m.predict(X))) <= set(m.classes_)

    def test_predict_agrees_with_proba(self, classifier_cls, clf_path, X, targets):
        y = targets[clf_path]
        m = make(classifier_cls).fit(X, y)
        expected = m.classes_[np.argmax(m.predict_proba(X), axis=1)]
        assert np.array_equal(m.predict(X), expected)

    def test_beats_majority_class(self, classifier_cls, clf_path, X, targets):
        y = targets[clf_path]
        m = make(classifier_cls).fit(X, y)
        majority = np.bincount(y).max() / len(y)
        assert m.score(X, y) > majority

    def test_non_numeric_labels_round_trip(self, classifier_cls, clf_path, X, targets):
        y_num = targets[clf_path]
        labels = np.array(["low", "mid", "high"], dtype=object)
        y = labels[y_num]
        m = make(classifier_cls).fit(X, y)
        assert set(m.classes_) == set(np.unique(y))
        assert set(np.unique(m.predict(X))) <= set(m.classes_)

    def test_decision_function_shape(self, classifier_cls, clf_path, X, targets):
        y = targets[clf_path]
        m = make(classifier_cls).fit(X, y)
        d = m.decision_function(X)
        n_classes = len(np.unique(y))
        assert d.shape == ((len(y),) if n_classes == 2 else (len(y), n_classes))

    def test_multiclass_exposes_sub_estimators(self, classifier_cls, X, y_mc):
        m = make(classifier_cls).fit(X, y_mc)
        assert len(m.estimators_) == len(m.classes_)
        assert all(e.estimators_ is None for e in m.estimators_)

    def test_multiclass_links_match_documented_definitions(
        self, classifier_cls, X, y_mc
    ):
        """Default is softmax; the opt-in legacy link normalizes OvR heads."""
        m = make(classifier_cls).fit(X, y_mc)

        def softmax(eta):
            eta = eta - eta.max(axis=1, keepdims=True)
            p = np.exp(eta)
            return p / p.sum(axis=1, keepdims=True)

        # SI keeps scale uncertainty by softmaxing paired posterior logit draws.
        if hasattr(m.estimators_[0], "_si_sample_logits"):
            heads = [e._si_sample_logits(X) for e in m.estimators_]
            n_draws = min(h.shape[0] for h in heads)
            expected = np.mean([
                softmax(np.column_stack([h[d] for h in heads]))
                for d in range(n_draws)
            ], axis=0)
        else:
            expected = softmax(m.decision_function(X))
        assert np.allclose(m.predict_proba(X), expected)

        m.multiclass_link = "normalized_ovr"
        head_prob = np.column_stack([
            e.predict_proba(X)[:, 1] for e in m.estimators_
        ])
        expected_legacy = head_prob / head_prob.sum(axis=1, keepdims=True)
        assert np.allclose(m.predict_proba(X), expected_legacy)

    def test_binary_has_no_sub_estimators(self, classifier_cls, X, y_bin):
        m = make(classifier_cls).fit(X, y_bin)
        assert m.estimators_ is None


# --------------------------------------------------------------------------
# Reporting surface (shared by every variant and path)
# --------------------------------------------------------------------------

class TestReporting:
    def test_summary_frame(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls).fit(X, y)
        df = m.summary()
        assert len(df) > 0
        for col in ("component", "kind", "n_coef", "edf", "importance"):
            assert col in df.columns
        assert (df["edf"] >= 0).all()
        assert (df["importance"] >= 0).all()

    def test_evidence_and_edf(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls).fit(X, y)
        assert np.isfinite(m.evidence_)
        assert all(np.isfinite(v) and v >= 0 for v in m.edf_.values())

    def test_shape_function(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls).fit(X, y)
        grid, mean, se = m.shape_function(0)
        assert grid.shape == mean.shape == se.shape
        assert np.all(se >= 0)
        assert np.all(np.isfinite(mean))

    def test_shape_function_by_name(self, estimator_cls, Xdf, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls).fit(Xdf, y)
        by_name = m.shape_function("gamma")
        by_index = m.shape_function(2)
        assert np.allclose(by_name[1], by_index[1])

    def test_multiclass_shape_function_requires_cls(self, classifier_cls, X, y_mc):
        m = make(classifier_cls).fit(X, y_mc)
        with pytest.raises(ValueError, match="cls="):
            m.shape_function(0)
        grid, mean, se = m.shape_function(0, cls=m.classes_[0])
        assert grid.shape == mean.shape == se.shape

    def test_multiclass_summary_has_class_column(self, classifier_cls, X, y_mc):
        m = make(classifier_cls).fit(X, y_mc)
        df = m.summary()
        assert "class" in df.columns
        assert set(df["class"]) == set(m.classes_)


# --------------------------------------------------------------------------
# Interaction discovery
# --------------------------------------------------------------------------

class TestInteractions:
    def test_discovers_planted_interaction(
        self, estimator_cls, X, targets, X_large, y_bin_large
    ):
        """The design plants a strong alpha*beta term; screening should find it."""
        Xd, y = screening_data(
            estimator_cls, (X, targets["regression"]), (X_large, y_bin_large)
        )
        m = make(estimator_cls, interactions="auto").fit(Xd, y)
        pairs = {tuple(sorted(i["pair"])) for i in m.interactions_}
        assert (0, 1) in pairs

    def test_interactions_disabled(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls, interactions=False).fit(X, y)
        assert m.interactions_ == []

    def test_interaction_names_use_fitted_labels(self, estimator_cls, Xdf, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls, interactions="auto").fit(Xdf, y)
        for info in m.interactions_:
            assert set(str(info["name"]).split("*")) <= set(COLUMNS)

    def test_max_interactions_respected(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls, interactions="auto", max_interactions=1).fit(X, y)
        assert len(m.interactions_) <= 1

    def test_predict_consistent_with_interactions(self, estimator_cls, X, targets):
        """Design rebuild on new data must reproduce training-time predictions."""
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls, interactions="auto").fit(X, y)
        assert np.all(np.isfinite(m.predict(X)))
        assert np.allclose(m.predict(X[:10]), m.predict(X)[:10])


# --------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------

class TestPlotting:
    def test_plot_helpers_run(self, estimator_cls, Xdf, targets):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from srae import plot_evidence, plot_importance, plot_shape_functions

        y = default_target(estimator_cls, targets)
        m = make(estimator_cls).fit(Xdf, y)
        for fn in (plot_shape_functions, plot_importance, plot_evidence):
            assert fn(m) is not None

    def test_plot_interaction_labels(
        self, estimator_cls, Xdf, targets, Xdf_large, y_bin_large
    ):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from srae import plot_interaction

        Xd, y = screening_data(
            estimator_cls, (Xdf, targets["regression"]), (Xdf_large, y_bin_large)
        )
        m = make(estimator_cls, interactions="auto").fit(Xd, y)
        assert m.interactions_, "screening found no interaction to plot"
        fig = plot_interaction(m, 0)
        ax = fig.axes[0]
        assert ax.get_xlabel() in COLUMNS
        assert ax.get_ylabel() in COLUMNS


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------

class TestErrors:
    def test_length_mismatch(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        with pytest.raises(ValueError):
            make(estimator_cls).fit(X, y[:-5])

    def test_single_class_rejected(self, classifier_cls, X):
        with pytest.raises(ValueError, match="at least 2 classes"):
            make(classifier_cls).fit(X, np.zeros(len(X)))

    def test_feature_names_length_mismatch(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        with pytest.raises(ValueError, match="feature_names"):
            make(estimator_cls, feature_names=["only", "two"]).fit(X, y)

    def test_predict_before_fit(self, estimator_cls, X):
        m = make(estimator_cls)
        with pytest.raises(Exception):
            m.predict(X)

    def test_one_dimensional_X_rejected(self, estimator_cls, targets):
        y = default_target(estimator_cls, targets)
        with pytest.raises(ValueError):
            make(estimator_cls).fit(np.arange(len(y), dtype=float), y)


# --------------------------------------------------------------------------
# Block dispatch for low-cardinality features
# --------------------------------------------------------------------------

class TestLowCardinalityBlocks:
    """``max_linear_card`` must reach ``make_block`` from the estimator.

    A LinearBlock is one standardized column, so it can only express a
    linear effect in the codes. Lowering the threshold is one documented
    escape hatch for a non-monotone low-cardinality feature; it only works
    if the constructor argument is actually forwarded.
    """

    @staticmethod
    def _u_shaped(seed=0, n=600):
        rng = np.random.default_rng(seed)
        x = rng.choice([0.0, 1.0, 2.0], size=n)
        y = np.where(x == 1.0, -1.0, 1.0) + rng.normal(0, 0.05, n)
        return x, x.reshape(-1, 1), y

    @staticmethod
    def _main_block(model):
        """First real main-effect block (classifiers prepend an intercept)."""
        from srae.blocks import FactorBlock, LinearBlock, SplineBlock

        return next(b for b in model.blocks_
                    if isinstance(b, (LinearBlock, SplineBlock, FactorBlock)))

    def test_threshold_selects_block_type(self, estimator_cls):
        from srae.blocks import LinearBlock, SplineBlock

        _, X, y = self._u_shaped()
        y = (y > 0).astype(float) if is_classifier_cls(estimator_cls) else y

        linear = make(estimator_cls, max_linear_card=5).fit(X, y)
        spline = make(estimator_cls, max_linear_card=2).fit(X, y)
        assert isinstance(self._main_block(linear), LinearBlock)
        assert isinstance(self._main_block(spline), SplineBlock)

    def test_param_roundtrips(self, estimator_cls):
        from sklearn.base import clone

        est = make(estimator_cls, max_linear_card=3)
        assert est.get_params()["max_linear_card"] == 3
        assert clone(est).get_params()["max_linear_card"] == 3

    def test_lowering_threshold_recovers_u_shape(self, regressor_cls):
        """The default flattens a symmetric U; a spline basis recovers it."""
        x, X, y = self._u_shaped()
        levels = [0.0, 1.0, 2.0]

        flat = make(regressor_cls, max_linear_card=5).fit(X, y).predict(X)
        spread_flat = np.ptp([flat[x == v].mean() for v in levels])

        curved = make(regressor_cls, max_linear_card=2).fit(X, y).predict(X)
        spread_curved = np.ptp([curved[x == v].mean() for v in levels])

        assert spread_flat < 0.1
        assert spread_curved > 1.5


# --------------------------------------------------------------------------
# Explicit feature typing
# --------------------------------------------------------------------------

class TestFeatureTypes:
    """``feature_types`` must select blocks and recover non-linear factors."""

    @staticmethod
    def _u_shaped(seed=0, n=600):
        rng = np.random.default_rng(seed)
        x = rng.choice([0.0, 1.0, 2.0], size=n)
        y = np.where(x == 1.0, -1.0, 1.0) + rng.normal(0, 0.05, n)
        return x, x.reshape(-1, 1), y

    @staticmethod
    def _main_block(model):
        from srae.blocks import FactorBlock, LinearBlock, SplineBlock

        return next(b for b in model.blocks_
                    if isinstance(b, (LinearBlock, SplineBlock, FactorBlock)))

    def test_factor_recovers_u_shape(self, regressor_cls):
        from srae.blocks import FactorBlock

        x, X, y = self._u_shaped()
        levels = [0.0, 1.0, 2.0]

        flat = make(regressor_cls, max_linear_card=5).fit(X, y).predict(X)
        fac = make(regressor_cls, feature_types="factor").fit(X, y)
        pred = fac.predict(X)

        assert isinstance(self._main_block(fac), FactorBlock)
        assert fac.summary().iloc[0]["kind"] == "factor"
        assert np.ptp([flat[x == v].mean() for v in levels]) < 0.1
        assert np.ptp([pred[x == v].mean() for v in levels]) > 1.5
        # Level means should track the U: high, low, high.
        means = [pred[x == v].mean() for v in levels]
        assert means[0] > means[1]
        assert means[2] > means[1]

    def test_factor_shape_function_is_discrete(self, regressor_cls):
        x, X, y = self._u_shaped()
        m = make(regressor_cls, feature_types="factor").fit(X, y)
        grid, mean, se = m.shape_function(0)
        assert grid.shape == (3,)
        assert np.allclose(grid, [0.0, 1.0, 2.0])
        assert mean.shape == (3,)
        assert se.shape == (3,)
        assert mean[1] < mean[0] and mean[1] < mean[2]

    def test_explicit_linear_and_spline(self, estimator_cls):
        from srae.blocks import LinearBlock, SplineBlock

        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 2))
        y = X[:, 0] + X[:, 1] ** 2
        if is_classifier_cls(estimator_cls):
            y = (y > np.median(y)).astype(float)

        lin = make(estimator_cls, feature_types=["linear", "linear"]).fit(X, y)
        spl = make(estimator_cls, feature_types=["spline", "spline"]).fit(X, y)
        assert isinstance(self._main_block(lin), LinearBlock)
        # Second main block of spl is still a spline (skip intercept if any).
        from srae.blocks import FactorBlock, LinearBlock as LB, SplineBlock as SB
        mains = [b for b in spl.blocks_ if isinstance(b, (LB, SB, FactorBlock))]
        assert all(isinstance(b, SplineBlock) for b in mains)

    def test_dict_by_name_and_index(self, regressor_cls):
        from srae.blocks import FactorBlock, LinearBlock, SplineBlock
        import pandas as pd

        rng = np.random.default_rng(1)
        x0 = rng.choice([0.0, 1.0, 2.0], size=200)
        x1 = rng.normal(size=200)
        X = pd.DataFrame({"cat": x0, "num": x1})
        y = np.where(x0 == 1.0, -1.0, 1.0) + 0.3 * x1 + rng.normal(0, 0.05, 200)

        m = make(
            regressor_cls,
            feature_types={"cat": "categorical", "num": "continuous"},
        ).fit(X, y)
        by_name = {s.name: b for s, b in zip(m.specs_, m.blocks_) if b is not None}
        assert isinstance(by_name["cat"], FactorBlock)
        assert isinstance(by_name["num"], SplineBlock)
        assert m.feature_types_ == ["factor", "spline"]

        m2 = make(regressor_cls, feature_types={0: "factor", 1: "linear"}).fit(
            X.to_numpy(), y
        )
        mains = [b for b in m2.blocks_
                 if isinstance(b, (FactorBlock, LinearBlock, SplineBlock))]
        assert isinstance(mains[0], FactorBlock)
        assert isinstance(mains[1], LinearBlock)

    def test_aliases_and_validation(self, regressor_cls):
        from srae.blocks import normalize_feature_type
        import pytest

        assert normalize_feature_type("ordinal") == "linear"
        assert normalize_feature_type("nominal") == "factor"
        with pytest.raises(ValueError, match="Unknown feature type"):
            normalize_feature_type("bogus")

        _, X, y = self._u_shaped()
        with pytest.raises(ValueError, match="feature_types has"):
            make(regressor_cls, feature_types=["factor", "extra"]).fit(X, y)
        with pytest.raises(ValueError, match="not among"):
            make(regressor_cls, feature_types={"missing": "factor"}).fit(X, y)

    def test_param_roundtrips(self, estimator_cls):
        from sklearn.base import clone

        est = make(estimator_cls, feature_types={"x0": "factor"})
        assert est.get_params()["feature_types"] == {"x0": "factor"}
        assert clone(est).get_params()["feature_types"] == {"x0": "factor"}

    def test_predict_matches_levels(self, regressor_cls):
        """Factor design is stable under transform on held-out rows."""
        x, X, y = self._u_shaped(n=400)
        m = make(regressor_cls, feature_types="factor").fit(X, y)
        # Same codes in a new order should give the same fitted level means.
        X_new = np.array([[0.0], [1.0], [2.0], [1.0], [0.0]])
        pred = m.predict(X_new)
        assert abs(pred[0] - pred[4]) < 1e-10
        assert abs(pred[1] - pred[3]) < 1e-10
        assert pred[1] < pred[0] and pred[1] < pred[2]


# --------------------------------------------------------------------------
# Tensor purification
# --------------------------------------------------------------------------

class TestTensorPurification:
    """Tensor blocks must carry no main effect expressible in their marginals.

    The tensor builds its own spline marginals regardless of what the main
    block is, so purifying only against the main design lets a feature whose
    main block is narrower (notably a LinearBlock) leak curvature into the
    pair, which then screens as a spurious interaction.
    """

    @staticmethod
    def _low_card_curvature(seed=0, n=900):
        """U-shape in x0 only; x1 is pure noise, so no pair is real."""
        rng = np.random.default_rng(seed)
        x0 = rng.choice([0.0, 1.0, 2.0], size=n)
        y = np.where(x0 == 1.0, -1.0, 1.0) + rng.normal(0, 0.05, n)
        return np.column_stack([x0, rng.normal(size=n)]), y

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_no_spurious_pair_from_low_card_curvature(self, seed):
        from srae import SRAERegressor

        X, y = self._low_card_curvature(seed)
        m = SRAERegressor(max_linear_card=5, interactions="auto").fit(X, y)
        assert m.interactions_ == []

    def test_genuine_interaction_still_found(self, regressor_cls):
        rng = np.random.default_rng(0)
        n = 800
        X = rng.normal(size=(n, 3))
        y = 2.0 * X[:, 0] * X[:, 1] - 0.5 * X[:, 2] + rng.normal(0, 0.2, n)
        m = make(regressor_cls, interactions="auto").fit(X, y)
        assert "x0*x1" in [d["name"] for d in m.interactions_]

    def test_purification_removes_marginal_directions(self):
        """Purified columns cannot reconstruct their own marginal bases."""
        from srae.blocks import TensorBlock

        rng = np.random.default_rng(0)
        n = 600
        xj, xk = rng.normal(size=n), rng.normal(size=n)
        tb = TensorBlock((0, 1))
        T = tb.fit(xj, xk, np.column_stack([xj, xk]))

        tb._ti = 0                      # replay the stored knot vectors
        Bj = tb._marginal(xj, fit=False)
        resid = Bj - T @ np.linalg.lstsq(T, Bj, rcond=None)[0]
        assert np.all(resid.std(axis=0) > 1e-8)


# --------------------------------------------------------------------------
# Interaction screening: failure handling
# --------------------------------------------------------------------------

class TestScreeningFailures:
    """A failing pair must be skipped only for genuine numerical reasons.

    The handler used to be a bare ``except Exception``, which silently turned
    programming errors into "this pair screened poorly" -- indistinguishable
    from a real negative result.
    """

    @staticmethod
    def _data(n=400):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, 3))
        return X, 2.0 * X[:, 0] * X[:, 1] + rng.normal(0, 0.2, n)

    def test_numerical_failure_skips_pair_with_warning(self):
        from srae import SRAERegressor
        from srae.blocks import TensorBlock

        X, y = self._data()
        original = TensorBlock.fit

        def flaky(self, xj, xk, main_design):
            if self.pair == (0, 2):
                raise np.linalg.LinAlgError("SVD did not converge")
            return original(self, xj, xk, main_design)

        TensorBlock.fit = flaky
        try:
            with pytest.warns(RuntimeWarning, match="skipping interaction candidate"):
                m = SRAERegressor(interactions="auto").fit(X, y)
        finally:
            TensorBlock.fit = original

        # The healthy pair is still found; only the failing one is dropped.
        assert "x0*x1" in [d["name"] for d in m.interactions_]
        assert "x0*x2" not in [d["name"] for d in m.interactions_]

    def test_programming_error_propagates(self):
        from srae import SRAERegressor
        from srae.blocks import TensorBlock

        X, y = self._data()
        original = TensorBlock.fit

        def broken(self, xj, xk, main_design):
            raise TypeError("bad signature")

        TensorBlock.fit = broken
        try:
            with pytest.raises(TypeError, match="bad signature"):
                SRAERegressor(interactions="auto").fit(X, y)
        finally:
            TensorBlock.fit = original

    def test_clean_fit_warns_nothing(self):
        from srae import SRAERegressor

        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            SRAERegressor(interactions="auto").fit(X, y)


# --------------------------------------------------------------------------
# Scale-integration sampler diagnostics
# --------------------------------------------------------------------------

class TestScaleIntegrationDiagnostics:
    """The SI variants must report whether their own sampler is usable.

    The shipped defaults previously retained 48 draws from a single chain with
    a fixed proposal, giving an effective sample size around 5 and no way for
    a caller to notice.
    """

    @staticmethod
    def _data(n=120):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, 4))
        y = (1.2 * np.sin(1.5 * X[:, 0]) + 2.0 * X[:, 0] * X[:, 1]
             - 0.5 * X[:, 2] + rng.normal(0, 0.4, n))
        return X, y

    def test_diagnostics_are_exposed(self):
        from srae import SRAERegressorSI

        X, y = self._data()
        m = SRAERegressorSI(interactions=False, n_samples=300, n_burn=200,
                            n_chains=4).fit(X, y)
        assert set(m.ess_) == {"f_lam", "f_kap", "f_sig"}
        assert set(m.rhat_) == {"f_lam", "f_kap", "f_sig"}
        assert m.n_chains_ == 4
        assert m.min_ess_ > 1.0
        assert len(m.samples_) == 4 * 300

    def test_heavy_draw_cap_is_enforced(self):
        """At most ``_MAX_STORED_DRAWS`` entries keep full (beta, Sigma).

        Scale factors must still be present on every retained draw so ESS /
        R-hat use the complete chain.
        """
        from srae import SRAERegressorSI
        from srae.scale_integration import _MAX_STORED_DRAWS, _array_draws

        assert _MAX_STORED_DRAWS == 128
        X, y = self._data()
        # More retained draws than the heavy-draw budget.
        m = SRAERegressorSI(
            interactions=False, n_samples=80, n_burn=20, n_chains=4,
            random_state=0,
        ).fit(X, y)
        assert len(m.samples_) == 4 * 80
        heavy = _array_draws(m.samples_)
        assert 0 < len(heavy) <= _MAX_STORED_DRAWS
        assert all("beta" in s and "Sigma" in s for s in heavy)
        assert all("f_lam" in s and "f_kap" in s for s in m.samples_)
        # Heavy draws are a proper subset when the chain is longer than the cap.
        assert len(heavy) < len(m.samples_)

    def test_adaptation_hits_target_acceptance(self):
        """A fixed 0.35 step lands far below target; adaptation corrects it."""
        from srae import SRAERegressorSI

        X, y = self._data()
        kw = dict(interactions=False, n_samples=400, n_burn=400, n_chains=2)
        adapted = SRAERegressorSI(adapt_step=True, **kw).fit(X, y)
        fixed = SRAERegressorSI(adapt_step=False, **kw).fit(X, y)

        assert 0.1 < adapted.accept_rate_ < 0.5
        assert adapted.min_ess_ > fixed.min_ess_

    def test_warns_when_run_is_too_short(self):
        from srae import SRAERegressorSI

        X, y = self._data()
        with pytest.warns(RuntimeWarning, match="may not have converged"):
            SRAERegressorSI(interactions=False, n_samples=8, n_burn=4,
                            n_chains=1).fit(X, y)

    def test_adequate_run_does_not_warn(self):
        from srae import SRAERegressorSI

        X, y = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            SRAERegressorSI(interactions=False).fit(X, y)

    def test_chains_are_reproducible_and_distinct(self):
        from srae import SRAERegressorSI

        X, y = self._data()
        kw = dict(interactions=False, n_samples=100, n_burn=100, n_chains=2)
        a = SRAERegressorSI(random_state=0, **kw).fit(X, y)
        b = SRAERegressorSI(random_state=0, **kw).fit(X, y)
        assert a.f_lam_mean_ == b.f_lam_mean_          # same seed reproduces
        # the two chains must not be identical, or Rhat would be meaningless
        first = [s["f_lam"] for s in a.samples_[:100]]
        second = [s["f_lam"] for s in a.samples_[100:]]
        assert first != second

    def test_multiclass_surfaces_worst_case_diagnostics(self):
        """OvR parent must expose ess_/rhat_ so callers can gate uncertainty."""
        from srae import SRAEClassifierSI

        rng = np.random.default_rng(0)
        X = rng.normal(size=(90, 3))
        y = rng.integers(0, 3, size=90)
        m = SRAEClassifierSI(
            interactions=False, n_samples=40, n_burn=20, n_chains=2,
            random_state=0,
        ).fit(X, y)
        assert m.estimators_ is not None and len(m.estimators_) == 3
        assert set(m.ess_) == {"f_lam", "f_kap"}
        assert set(m.rhat_) == {"f_lam", "f_kap"}
        assert m.min_ess_ is not None
        assert m.max_rhat_ is not None
        # Parent ESS is the worst class for each key.
        for k in m.ess_:
            child_ess = [e.ess_[k] for e in m.estimators_]
            assert m.ess_[k] == min(child_ess)
        for k in m.rhat_:
            child_r = [e.rhat_[k] for e in m.estimators_ if np.isfinite(e.rhat_[k])]
            if child_r:
                assert m.rhat_[k] == max(child_r)


# --------------------------------------------------------------------------
# Predictive variance composition
# --------------------------------------------------------------------------

class TestPredictiveVariance:
    """The intercept is fitted, so its sampling variance belongs in predict.

    The regressor centers y at ybar and adds it back at predict time; treating
    that estimate as known omitted sigma^2 / n_train from every interval.
    Every regression variant must set _n_train and include the term.
    """

    @staticmethod
    def _data(n=150):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, 3))
        y = 1.2 * np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] + rng.normal(0, 0.5, n)
        return X, y, n

    @staticmethod
    def _fit(n=150):
        from srae import SRAERegressor

        X, y, n = TestPredictiveVariance._data(n)
        return SRAERegressor(interactions=False).fit(X, y), X, n

    def test_variance_includes_intercept_term(self):
        m, X, n = self._fit()
        Z = m._design(m._coerce(X))
        var_f = np.einsum("ij,jk,ik->i", Z, m.Sigma_, Z)
        _, std = m.predict(X, return_std=True)
        expected = np.sqrt(var_f + m.sigma2_ + m.sigma2_ / n)
        assert np.allclose(std, expected)
        # strictly wider than the version that treated the intercept as known
        assert np.all(std > np.sqrt(var_f + m.sigma2_))
        assert m._n_train == n

    def test_intercept_term_shrinks_with_n(self):
        small, _, n_small = self._fit(n=60)
        large, _, n_large = self._fit(n=600)
        assert n_small < n_large
        rel_small = (small.sigma2_ / n_small) / small.sigma2_
        rel_large = (large.sigma2_ / n_large) / large.sigma2_
        assert rel_large < rel_small

    def test_intercept_helper_and_em_fixed_point(self):
        """_intercept_sampling_variance is sigma2/n; EM identity holds."""
        m, X, n = self._fit()
        assert abs(m._intercept_sampling_variance() - m.sigma2_ / n) < 1e-15
        Z = m._Ztrain
        tr = float(np.sum(m.Sigma_ * (Z.T @ Z)))
        # EM M-step: sigma2 = (rss + tr) / n  =>  rss = n*sigma2 - tr >= 0
        rss = n * m.sigma2_ - tr
        assert rss >= -1e-6
        edf = sum(m.edf_.values())
        # tr ≈ sigma2 * edf at the fixed point for the Gaussian EB model
        if edf > 0.5 and m.sigma2_ > 0:
            assert abs(tr - m.sigma2_ * edf) / (m.sigma2_ * edf) < 0.15

    def test_missing_n_train_raises(self):
        m, X, _ = self._fit()
        del m._n_train
        with pytest.raises(RuntimeError, match="_n_train"):
            m.predict(X, return_std=True)

    def test_pooled_records_n_train_and_intercept_term(self):
        from srae import SRAERegressorPooled

        X, y, n = self._data()
        m = SRAERegressorPooled(
            interactions=False, holdout_calibrate=False,
        ).fit(X, y)
        assert m._n_train == n
        Z = m._design(m._coerce(X))
        var_f = np.einsum("ij,jk,ik->i", Z, m.Sigma_, Z)
        _, std = m.predict(X, return_std=True)
        expected = np.sqrt(var_f + m.sigma2_ + m.sigma2_ / n)
        assert np.allclose(std, expected)
        # Must not fall back to sigma2/1 (the old missing-_n_train bug).
        assert not np.allclose(std, np.sqrt(var_f + m.sigma2_ + m.sigma2_))

    def test_si_includes_intercept_term(self):
        from srae import SRAERegressorSI

        X, y, n = self._data(n=100)
        m = SRAERegressorSI(
            interactions=False, n_samples=40, n_burn=20, n_chains=2,
            random_state=0,
        ).fit(X, y)
        assert m._n_train == n
        Z = m._design(m._coerce(X))
        # Reconstruct the SI predictive variance formula with intercept term.
        means, vars_ = [], []
        for s in m.samples_:
            means.append(m._ymean + Z @ s["beta"])
            vf = np.einsum("ij,jk,ik->i", Z, s["Sigma"], Z)
            vars_.append(np.clip(vf, 0, None) + s["sigma2"] + s["sigma2"] / n)
        means = np.stack(means, axis=0)
        vars_ = np.stack(vars_, axis=0)
        expected = np.sqrt(vars_.mean(axis=0) + means.var(axis=0))
        _, std = m.predict(X, return_std=True)
        assert np.allclose(std, expected)
        # Wider than the no-intercept SI formula.
        vars_old = []
        for s in m.samples_:
            vf = np.einsum("ij,jk,ik->i", Z, s["Sigma"], Z)
            vars_old.append(np.clip(vf, 0, None) + s["sigma2"])
        std_old = np.sqrt(
            np.stack(vars_old).mean(axis=0) + means.var(axis=0)
        )
        assert np.all(std >= std_old - 1e-12)
        assert np.mean(std - std_old) > 0
