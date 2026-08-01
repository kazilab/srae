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
        """Each variant must use the link its documentation claims.

        Since 0.0.10 the default is ``joint``: the Laplace posterior of the
        joint multinomial refit, moderated toward ``1/K``.  The pooled variant
        overrides the multiclass fit and has no joint analogue, so it falls
        back to ``normalized_ovr`` -- deliberately, not to ``softmax``, which
        measured worse on both log-loss and ECE.  The scale-integrated variants
        keep their own paired-draw route.
        """
        m = make(classifier_cls).fit(X, y_mc)

        def softmax(eta):
            eta = eta - eta.max(axis=1, keepdims=True)
            p = np.exp(eta)
            return p / p.sum(axis=1, keepdims=True)

        head_prob = np.column_stack([
            e.predict_proba(X)[:, 1] for e in m.estimators_
        ])
        normalized = head_prob / head_prob.sum(axis=1, keepdims=True)

        if hasattr(m.estimators_[0], "_si_sample_logits"):
            # SI keeps scale uncertainty by softmaxing paired posterior draws.
            heads = [e._si_sample_logits(X) for e in m.estimators_]
            n_draws = min(h.shape[0] for h in heads)
            expected = np.mean([
                softmax(np.column_stack([h[d] for h in heads]))
                for d in range(n_draws)
            ], axis=0)
        elif getattr(m, "_joint_", None) is not None:
            Eta, vbar = m._joint_logits_and_variance(X)
            expected = softmax(Eta / np.sqrt(1.0 + (np.pi / 8.0) * vbar)[:, None])
        else:
            expected = normalized
        assert np.allclose(m.predict_proba(X), expected)

        m.multiclass_link = "normalized_ovr"
        assert np.allclose(m.predict_proba(X), normalized)

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
# Joint multinomial multiclass
# --------------------------------------------------------------------------

class TestJointMultinomial:
    """The multiclass posterior must be joint, not a stack of binary fits.

    Independent one-vs-rest fits leave the cross-class blocks of the Hessian at
    zero, so they carry no covariance between class surfaces and no neutral
    point to moderate toward.  0.0.10 refits the discovered structure under one
    softmax likelihood, which supplies both.
    """

    @staticmethod
    def _data(n=300, K=4, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 4))
        lg = np.column_stack([1.2 * np.sin(1.3 * X[:, 0] + c)
                              + 0.7 * X[:, 2] for c in range(K)])
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        y = np.array([rng.choice(K, p=pi) for pi in p])
        return X, y

    def test_engine_reduces_to_the_logistic_engine_at_two_classes(self):
        """The strongest available check: at K=2 the multinomial model *is*
        the logistic one, so both engines must agree bit for bit."""
        from srae.inference import BlockSpec, fit_logistic_eb, fit_multinomial_eb

        rng = np.random.default_rng(0)
        n = 300
        Z = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
        eta = Z @ np.array([0.2, 1.0, -0.7, 0.5])
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
        s = np.array([0.0, 1.0, 1.0, 1.0])

        mn = fit_multinomial_eb(Z, np.column_stack([1 - y, y]),
                                [BlockSpec("f", slice(0, 4), s)],
                                max_iter=300, tol=1e-11)
        lg = fit_logistic_eb(Z, y, [BlockSpec("f", slice(0, 4), s)],
                             max_iter=300, tol=1e-11)
        assert mn["evidence"] == pytest.approx(lg["evidence"], abs=1e-8)
        assert mn["edf"]["f"] == pytest.approx(lg["edf"]["f"], rel=1e-8)

        # Coefficients live in the sum-to-zero contrast basis, so they carry a
        # fixed scale factor (sqrt(2) at K=2). The invariant quantity is the
        # logit contrast the model actually predicts with.
        C = mn["contrasts"]
        contrast = float(C[1, 0] - C[0, 0]) * mn["beta"][0]
        assert np.abs(contrast - lg["beta"]).max() < 1e-8

    def test_gradient_and_hessian_match_finite_differences(self):
        """Guards the analytic derivatives that the Newton solve depends on."""
        from srae.inference import (_contrast_basis, _multinomial_hessian,
                                    _softmax_rows)

        rng = np.random.default_rng(0)
        n, p, K = 150, 5, 4
        Z = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
        C = _contrast_basis(K)
        P0 = _softmax_rows(Z @ (rng.normal(size=(K - 1, p)) * 0.8).T @ C.T)
        y = np.array([rng.choice(K, p=pi) for pi in P0])
        Y = np.eye(K)[y]
        a = np.full((K - 1) * p, 0.7)

        def grad(b):
            Q = _softmax_rows(Z @ b.reshape(K - 1, p).T @ C.T)
            return (C.T @ ((Y - Q).T @ Z)).ravel() - a * b

        b0 = rng.normal(size=(K - 1) * p) * 0.3
        P = _softmax_rows(Z @ b0.reshape(K - 1, p).T @ C.T)
        H = _multinomial_hessian(Z, P, C, a, K - 1)
        step = 1e-6 * np.eye(len(b0))
        Hfd = np.column_stack([(grad(b0 + e) - grad(b0 - e)) / 2e-6
                               for e in step])
        assert np.abs(H + Hfd).max() < 1e-5 * np.abs(H).max()

    def test_fit_exposes_a_joint_posterior(self):
        """``joint_`` carries K-1 contrasts against a reference class."""
        from srae import SRAEClassifier

        X, y = self._data()
        m = SRAEClassifier(interactions=False, n_knots=6).fit(X, y)
        K = len(m.classes_)
        assert m.joint_["beta"].shape[0] == K - 1
        C = m.joint_["contrasts"]
        assert C.shape == (K, K - 1)
        assert np.allclose(C.T @ C, np.eye(K - 1), atol=1e-10)   # orthonormal
        assert np.abs(C.sum(axis=0)).max() < 1e-10               # sum-to-zero
        p = m.joint_["beta"].shape[1]
        assert m.joint_["Sigma"].shape == ((K - 1) * p,) * 2
        # the cross-class blocks are what one-vs-rest cannot produce
        off = m.joint_["Sigma"][:p, p:2 * p]
        assert np.abs(off).max() > 0

    def test_probabilities_are_moderated_toward_uniform(self):
        """Moderation must pull toward 1/K, never past it or away from it."""
        from srae import SRAEClassifier

        X, y = self._data()
        m = SRAEClassifier(interactions=False, n_knots=6).fit(X, y)
        K = len(m.classes_)
        P = m.predict_proba(X)
        assert np.allclose(P.sum(1), 1.0)

        Eta, vbar = m._joint_logits_and_variance(X)
        assert np.all(vbar >= 0)
        unmoderated = np.exp(Eta - Eta.max(1, keepdims=True))
        unmoderated /= unmoderated.sum(1, keepdims=True)
        # every row is at least as close to uniform as the unmoderated one
        d_mod = np.abs(P - 1.0 / K).sum(1)
        d_raw = np.abs(unmoderated - 1.0 / K).sum(1)
        assert np.all(d_mod <= d_raw + 1e-9)

    def test_moderation_is_independent_of_the_reference_class(self):
        """A common per-row factor keeps the link reference-invariant.

        Scaling every logit by one number commutes with the constant shift that
        relabelling the reference introduces, which a per-class factor would
        not.  Checked by relabelling the classes cyclically and refitting.
        """
        from srae import SRAEClassifier

        X, y = self._data()
        base = SRAEClassifier(interactions=False, n_knots=6).fit(X, y)
        K = len(base.classes_)
        rolled = SRAEClassifier(interactions=False, n_knots=6).fit(X, (y + 1) % K)
        # rolled label j corresponds to original class j-1, so undo by taking
        # column (c+1) % K for original class c.
        P0 = base.predict_proba(X)
        P1 = rolled.predict_proba(X)[:, (np.arange(K) + 1) % K]
        assert np.abs(P0 - P1).max() < 1e-6

    def test_legacy_links_remain_reproducible(self):
        """Both retired routes stay available for published results."""
        from srae import SRAEClassifier

        class Softmax(SRAEClassifier):
            multiclass_link = "softmax"

        class NormOvR(SRAEClassifier):
            multiclass_link = "normalized_ovr"

        X, y = self._data()
        kw = dict(interactions=False, n_knots=6)
        Pj = SRAEClassifier(**kw).fit(X, y).predict_proba(X)
        Ps = Softmax(**kw).fit(X, y).predict_proba(X)
        Pn = NormOvR(**kw).fit(X, y).predict_proba(X)
        for P in (Pj, Ps, Pn):
            assert np.allclose(P.sum(1), 1.0)
        assert np.abs(Pj - Ps).max() > 1e-3
        assert np.abs(Pj - Pn).max() > 1e-3

    def test_binary_path_is_untouched(self):
        """K = 2 must not acquire a joint fit or change behaviour."""
        from srae import SRAEClassifier

        X, y = self._data()
        m = SRAEClassifier(interactions=False, n_knots=6).fit(X, (y > 1).astype(int))
        assert m.estimators_ is None
        assert not hasattr(m, "joint_")
        assert np.allclose(m.predict_proba(X).sum(1), 1.0)

    def test_oversized_problems_decline_the_joint_fit_and_say_so(self):
        """The Hessian is ((K-1) * n_columns) square, so it must be bounded.

        sklearn's ``check_estimator`` fits a 200-class problem that would ask
        for a 14129-square Hessian -- 1.6 GB per matrix, and enough of them to
        be OOM-killed. Declining must leave a working model on the legacy link
        and warn, not raise and not silently degrade.
        """
        import srae.model as model_mod
        from srae import SRAEClassifier

        X, y = self._data(n=200)
        original = model_mod._MAX_JOINT_DIM
        try:
            model_mod._MAX_JOINT_DIM = 4        # force the limit
            with pytest.warns(RuntimeWarning, match="joint multinomial refit"):
                m = SRAEClassifier(interactions=False, n_knots=5).fit(X, y)
        finally:
            model_mod._MAX_JOINT_DIM = original

        assert getattr(m, "_joint_", None) is None
        P = m.predict_proba(X)
        assert np.allclose(P.sum(1), 1.0)
        ref = np.column_stack([e.predict_proba(X)[:, 1] for e in m.estimators_])
        assert np.allclose(P, ref / ref.sum(1, keepdims=True))

    def test_variants_without_a_joint_fit_fall_back_to_normalized_ovr(self):
        """Pooled / SI override the multiclass fit and have no joint analogue.

        Their fallback must be ``normalized_ovr``, the better-calibrated of the
        two legacy routes, not the ``softmax`` it replaced.
        """
        from srae import SRAEClassifierPooled

        X, y = self._data(n=200)
        m = SRAEClassifierPooled(interactions=False, n_knots=5).fit(X, y)
        assert getattr(m, "_joint_", None) is None
        P = m.predict_proba(X)
        ref = np.column_stack([e.predict_proba(X)[:, 1] for e in m.estimators_])
        assert np.allclose(P, ref / ref.sum(1, keepdims=True))


# --------------------------------------------------------------------------
# Spline roughness penalty
# --------------------------------------------------------------------------

class TestRoughnessPenalty:
    """The penalty must be the integral it claims to be, on any knot spacing.

    SRAE places knots at empirical quantiles, so the plain coefficient
    difference penalty of Eilers and Marx -- formulated for equally spaced
    knots -- does not represent a roughness measure here.  0.0.7 replaced it
    with the exact integrated squared second derivative.
    """

    @staticmethod
    def _skewed(n=300, seed=0):
        """Gamma-distributed x: quantile knots come out strongly non-uniform."""
        rng = np.random.default_rng(seed)
        x = np.sort(rng.gamma(2.0, 1.0, n))
        y = np.sin(1.2 * x) + rng.normal(0, 0.2, n)
        return x, y - y.mean()

    def test_quadratic_form_is_the_exact_integral(self):
        """beta' Omega beta == integral (f'')^2 dx, checked against quadrature."""
        from scipy.integrate import quad
        from scipy.interpolate import BSpline

        from srae.blocks import SplineBlock, _integral_derivative_penalty

        x, _ = self._skewed()
        blk = SplineBlock(n_knots=10)
        blk.fit(x)
        t, k = blk.t_, blk.degree
        Om = _integral_derivative_penalty(t, k, order=2)

        rng = np.random.default_rng(1)
        for _ in range(3):
            c = rng.normal(size=len(t) - k - 1)
            exact = quad(lambda u: BSpline(t, c, k).derivative(2)(u) ** 2,
                         t[k], t[-k - 1], limit=400)[0]
            assert abs(float(c @ Om @ c) - exact) < 1e-6 * abs(exact)

    def test_knots_are_non_uniform_enough_to_matter(self):
        """Guards the premise: on quantile knots the spacing really does vary."""
        from srae.blocks import SplineBlock

        x, _ = self._skewed()
        blk = SplineBlock(n_knots=10)
        blk.fit(x)
        spacing = np.diff(blk.t_[blk.degree:-blk.degree])
        assert spacing.max() / spacing.min() > 10

    def test_null_space_is_exactly_the_straight_lines(self):
        """Order-2 roughness annihilates linear functions and nothing else.

        Under the difference penalty on quantile knots the null space was only
        *trend-like*, which is what forced the hedged wording about ``kappa_j``
        in the docs before 0.0.7.
        """
        from srae.blocks import (SplineBlock, _bspline_design,
                                 _integral_derivative_penalty)

        x, _ = self._skewed()
        blk = SplineBlock(n_knots=10)
        blk.fit(x)
        Om = _integral_derivative_penalty(blk.t_, blk.degree, order=2)
        w, V = np.linalg.eigh(Om)
        null = V[:, w < 1e-8 * w.max()]
        assert null.shape[1] == 2

        grid = np.linspace(blk.t_[blk.degree], blk.t_[-blk.degree - 1], 400)
        basis = _bspline_design(grid, blk.t_, blk.degree)
        line = np.column_stack([np.ones_like(grid), grid])
        for i in range(null.shape[1]):
            f = basis @ null[:, i]
            resid = f - line @ np.linalg.lstsq(line, f, rcond=None)[0]
            assert np.abs(resid).max() < 1e-9 * max(f.max() - f.min(), 1e-12)

    @pytest.mark.parametrize("dist", ["uniform", "normal", "gamma", "lognormal"])
    def test_null_space_is_one_canonical_column(self, dist):
        """One zero-penalty column, not two collinear ones.

        The order-2 penalty annihilates a two-dimensional space, but centering
        removes the constant fitted contribution, so exactly one function
        survives.  An eigensolver returns an arbitrary mixture of the two
        zero-eigenvalue vectors -- in practice always one leaving both columns
        non-constant -- so before 0.0.9 the block kept two perfectly collinear
        columns for that single function and reported ``n_coef`` one too high.
        """
        from srae.blocks import SplineBlock

        rng = np.random.default_rng(0)
        n = 300
        x = np.sort({"uniform": lambda: rng.uniform(0, 6, n),
                     "normal": lambda: rng.normal(3, 1, n),
                     "gamma": lambda: rng.gamma(2.0, 1.0, n),
                     "lognormal": lambda: rng.lognormal(0.4, 0.9, n)}[dist]())

        blk = SplineBlock(n_knots=10)
        Z = blk.fit(x)
        assert int(np.sum(blk.s_ == 0)) == 1
        assert Z.shape[1] == blk.s_.size

        # and no two columns are collinear
        C = np.corrcoef(Z.T)
        np.fill_diagonal(C, 0.0)
        assert np.abs(C).max() < 1.0 - 1e-6

    def test_canonicalizing_the_null_space_changes_no_fitted_quantity(self):
        """It is a representation change: only ``n_coef`` and ``kappa`` move.

        Two collinear coordinates sharing an isotropic ``kappa_j`` telescope to
        the same EM fixed point as one, so evidence, edf and the fitted
        function are untouched.  Compared here against the redundant
        parametrization built by hand.
        """
        from scipy.linalg import eigh

        from srae.blocks import SplineBlock, _bspline_design
        from srae.inference import BlockSpec, fit_gaussian_eb

        x, y = self._skewed()
        blk = SplineBlock(n_knots=10)
        Z = blk.fit(x)

        # rebuild without the canonicalization: rotate, then filter on norms
        B = _bspline_design(x, blk.t_, blk.degree)
        Bc = B - B.mean(axis=0)
        s, U = eigh(blk._penalty_matrix(B.shape[1]))
        s = np.clip(s, 0.0, None) / max(np.clip(s, 0.0, None).max(), 1e-300)
        s[s < 1e-10] = 0.0
        Zt = Bc @ U
        nr = np.linalg.norm(Zt, axis=0)
        keep = nr > 1e-8 * nr.max()
        Zt, s_red = Zt[:, keep], s[keep]
        sc = Zt.std(axis=0)
        sc[sc < 1e-12] = 1.0
        Zred, s_red = Zt / sc, s_red / sc**2

        assert Zred.shape[1] == Z.shape[1] + 1      # the redundant column
        assert int(np.sum(s_red == 0)) == 2

        def run(Zx, sx):
            b = [BlockSpec("f", slice(0, Zx.shape[1]), sx)]
            o = fit_gaussian_eb(Zx, y, b, max_iter=800, tol=1e-11)
            return o["evidence"], o["edf"]["f"], b[0].lam, Zx @ o["beta"]

        # Tolerances reflect the EM stopping rule, not the identity: the two
        # parametrizations have different coordinate counts, so EM halts at
        # marginally different points and the small difference in lambda
        # propagates to edf. Measured residuals are ~1e-3 nats of evidence --
        # against a 4.0-nat selection threshold -- and ~2e-4 relative on edf.
        new, old = run(Z, blk.s_), run(Zred, s_red)
        assert new[0] == pytest.approx(old[0], abs=5e-3)      # evidence, nats
        assert new[1] == pytest.approx(old[1], rel=1e-3)      # edf
        assert new[2] == pytest.approx(old[2], rel=1e-3)      # lambda
        assert np.abs(new[3] - old[3]).max() < 1e-3           # fitted values

    def test_kappa_direction_is_linear_in_raw_x(self):
        """The consequence users see: unpenalized columns are straight lines."""
        from srae.blocks import SplineBlock

        x, _ = self._skewed()
        blk = SplineBlock(n_knots=10)
        Z = blk.fit(x)
        line = np.column_stack([np.ones_like(x), x])
        unpenalized = np.where(blk.s_ == 0)[0]
        assert len(unpenalized) >= 1
        for i in unpenalized:
            z = Z[:, i]
            resid = z - line @ np.linalg.lstsq(line, z, rcond=None)[0]
            assert np.abs(resid).max() < 1e-8 * (z.max() - z.min())

    def test_integral_penalty_beats_differences_on_skewed_knots(self):
        """Why the default changed, pinned as a measurement.

        Same basis, same data, same engine -- only the penalty differs.  The
        difference penalty under-penalizes curvature where knots are dense, so
        it buys a worse marginal likelihood while spending more edf.
        """
        from srae.blocks import SplineBlock
        from srae.inference import BlockSpec, fit_gaussian_eb

        class Legacy(SplineBlock):
            penalty = "difference"

        x, y = self._skewed()

        def run(cls):
            blk = cls(n_knots=10)
            Z = blk.fit(x)
            b = [BlockSpec("f", slice(0, Z.shape[1]), blk.s_)]
            out = fit_gaussian_eb(Z, y, b, max_iter=800, tol=1e-11)
            return out["evidence"], out["edf"]["f"]

        ev_new, edf_new = run(SplineBlock)
        ev_old, edf_old = run(Legacy)
        assert ev_new > ev_old + 5.0
        assert edf_new < edf_old

    @pytest.mark.parametrize("scale", [1e-4, 1e-2, 1e2, 1e4])
    def test_fit_is_invariant_to_the_units_of_x(self, scale):
        """Rescaling a feature must not change anything at all.

        The roughness penalty carries units of ``x**-3``, so its eigenvalues
        move by nine orders of magnitude between millimetres and metres. The
        block normalizes them for exactly this reason: without it the EM update
        for ``lambda`` -- which starts at 1 -- lands in a degenerate fixed point
        where the prior already dominates, the update returns ``lambda``
        unchanged, and the component collapses to a straight line no matter
        what the data says.
        """
        from srae import SRAERegressor

        rng = np.random.default_rng(0)
        n = 400
        x = rng.normal(size=n)
        y = np.sin(1.5 * x) + rng.normal(0, 0.25, n)

        base = SRAERegressor(interactions=False).fit(x.reshape(-1, 1), y)
        moved = SRAERegressor(interactions=False).fit((scale * x).reshape(-1, 1), y)
        b, m = base.summary().iloc[0], moved.summary().iloc[0]

        assert m["edf"] == pytest.approx(b["edf"], rel=1e-6)
        assert m["lam"] == pytest.approx(b["lam"], rel=1e-6)
        assert m["kappa"] == pytest.approx(b["kappa"], rel=1e-6)
        assert moved.evidence_ == pytest.approx(base.evidence_, rel=1e-9)

    def test_difference_penalty_escape_hatch_restores_old_design(self):
        """``penalty='difference'`` must reproduce the pre-0.0.7 construction."""
        from srae.blocks import (SplineBlock, _difference_penalty,
                                 _integral_derivative_penalty)

        class Legacy(SplineBlock):
            penalty = "difference"

        x, _ = self._skewed()
        legacy, current = Legacy(n_knots=10), SplineBlock(n_knots=10)
        legacy.fit(x)
        current.fit(x)
        n_basis = len(legacy.t_) - legacy.degree - 1

        assert np.allclose(legacy._penalty_matrix(n_basis),
                           _difference_penalty(n_basis, order=2))
        assert np.allclose(
            current._penalty_matrix(n_basis),
            _integral_derivative_penalty(current.t_, current.degree, order=2))
        # The two knot vectors are identical, so any difference is the penalty.
        assert np.allclose(legacy.t_, current.t_)
        assert not np.allclose(legacy._penalty_matrix(n_basis),
                               current._penalty_matrix(n_basis))


# --------------------------------------------------------------------------
# Tensor blocks: basis invariance of the screening gain
# --------------------------------------------------------------------------

class TestTensorPenalty:
    """The screening gain must depend on the surface, not on its coordinates.

    Before 0.0.8 the tensor block carried an isotropic ridge on B-spline
    coefficients.  A ridge is not a functional of the fitted surface, so
    reparametrizing the marginals over the same function space moved the gain
    -- by more than the 4.0-nat selection threshold in the worst case measured.
    0.0.8 replaced it with the integrated squared second derivatives of the
    surface, which is such a functional and therefore transforms correctly.
    """

    @staticmethod
    def _setup(seed=0, n=400):
        from srae.blocks import SplineBlock

        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2))
        y = 2.0 * X[:, 0] * X[:, 1] + np.sin(1.5 * X[:, 0]) + rng.normal(0, 0.3, n)
        main = np.column_stack(
            [SplineBlock(n_knots=10).fit(X[:, j]) for j in range(2)]
        )
        return X, y - y.mean(), main

    @staticmethod
    def _run(X, y, main, rule, W=None, R=None, S=None):
        """Build the block by hand so the marginal basis can be swapped.

        ``rule`` selects which penalty is applied *in the coordinates actually
        used*: a ridge is the identity whatever the basis, while the roughness
        penalty of the same surfaces expressed through ``B R`` is
        ``(R kron S)' Omega (R kron S)``.
        """
        from srae.blocks import TensorBlock
        from srae.inference import BlockSpec, fit_gaussian_eb

        n = len(X)
        tb = TensorBlock(pair=(0, 1))
        tb._ts = []
        Bj = tb._marginal(X[:, 0], fit=True, side=0)
        Bk = tb._marginal(X[:, 1], fit=True, side=1)
        Om = tb._penalty_matrix(Bj.shape[1], Bk.shape[1])
        T = np.einsum("ij,ik->ijk", Bj, Bk).reshape(n, -1)
        if W is not None:
            T, Bj, Bk = T @ W, Bj @ R, Bk @ S
            Om = W.T @ Om @ W if rule == "roughness" else np.eye(T.shape[1])
        elif rule == "ridge":
            Om = np.eye(T.shape[1])

        P = tb._purify_design(Bj, Bk, main)
        Tp = T - P @ np.linalg.lstsq(P, T, rcond=None)[0]
        s, U = np.linalg.eigh(Om)
        s = np.clip(s, 0.0, None)
        keep = s > 1e-9 * s.max()
        Z, s = (Tp @ U)[:, keep], s[keep] / s[keep].max()
        sc = Z.std(axis=0)
        sc[sc < 1e-12] = 1.0
        blocks = [BlockSpec("t", slice(0, int(keep.sum())), s / sc**2)]
        out = fit_gaussian_eb(Z / sc, y, blocks, max_iter=1200, tol=1e-12)
        return out["evidence"], out["edf"]["t"], int(keep.sum())

    @staticmethod
    def _reparam(rng, p=5, q=5):
        R = np.eye(p) + 0.4 * rng.normal(size=(p, p))
        S = np.eye(q) + 0.4 * rng.normal(size=(q, q))
        return np.kron(R, S), R, S

    def test_penalty_is_the_exact_double_integral(self):
        """Check the Kronecker form against a dense independent quadrature."""
        from scipy.interpolate import BSpline

        from srae.blocks import (TensorBlock, _derivative_coef_operator,
                                 _marginal_penalty_parts,
                                 _tensor_roughness_penalty)

        rng = np.random.default_rng(0)
        tb = TensorBlock(pair=(0, 1))
        tb._ts = []
        Bj = tb._marginal(rng.normal(size=300), fit=True, side=0)
        Bk = tb._marginal(rng.normal(size=300), fit=True, side=1)
        tj, tk, k = tb._ts[0], tb._ts[1], tb.degree
        p, q = Bj.shape[1], Bk.shape[1]
        tjn = (tj - tj[0]) / (tj[-1] - tj[0])
        tkn = (tk - tk[0]) / (tk[-1] - tk[0])
        Om = _tensor_roughness_penalty(
            _marginal_penalty_parts(tjn, k, p, False),
            _marginal_penalty_parts(tkn, k, q, False))

        def design(t, order, grid):
            if order == 0:
                return BSpline.design_matrix(grid, t, k).toarray()
            D, t2, k2 = _derivative_coef_operator(t, k, order)
            return BSpline.design_matrix(grid, t2, k2).toarray() @ D

        gu = np.linspace(tjn[k], tjn[-k - 1], 1201)
        gv = np.linspace(tkn[k], tkn[-k - 1], 1201)
        Aj = [design(tjn, d, gu) for d in (0, 1, 2)]
        Ak = [design(tkn, d, gv) for d in (0, 1, 2)]
        for _ in range(2):
            c = rng.normal(size=p * q)
            M = c.reshape(p, q)
            F = ((Aj[2] @ M @ Ak[0].T) ** 2 + 2 * (Aj[1] @ M @ Ak[1].T) ** 2
                 + (Aj[0] @ M @ Ak[2].T) ** 2)
            grid = np.trapezoid(np.trapezoid(F, gv, axis=1), gu)
            assert abs(float(c @ Om @ c) - grid) < 5e-3 * abs(grid)

    def test_null_space_is_the_affine_surfaces(self):
        """Order-2 roughness annihilates ``a + b x_j + c x_k`` and nothing more.

        The bilinear ``x_j x_k`` must *not* be in it -- it is the simplest
        genuine interaction, and the mixed-derivative term is what penalizes it.
        """
        from srae.blocks import (TensorBlock, _marginal_penalty_parts,
                                 _tensor_roughness_penalty)

        tb = TensorBlock(pair=(0, 1))
        tb._ts = []
        rng = np.random.default_rng(0)
        tb._marginal(rng.normal(size=300), fit=True, side=0)
        tb._marginal(rng.normal(size=300), fit=True, side=1)
        norm = [(t - t[0]) / (t[-1] - t[0]) for t in tb._ts]
        Om = _tensor_roughness_penalty(
            *[_marginal_penalty_parts(t, tb.degree, 5, False) for t in norm])

        w = np.linalg.eigvalsh(Om)
        assert int(np.sum(w <= 1e-9 * w.max())) == 3      # 1, x_j, x_k

        # the bilinear surface, in coefficients: linear in each margin
        from srae.blocks import _bspline_design
        cs = []
        for t in norm:
            g = np.linspace(t[tb.degree], t[-tb.degree - 1], 200)
            cs.append(np.linalg.lstsq(_bspline_design(g, t, tb.degree),
                                      g, rcond=None)[0])
        bilinear = np.kron(cs[0], cs[1])
        assert float(bilinear @ Om @ bilinear) > 1e-6

    def test_gain_is_invariant_to_the_marginal_basis(self):
        """The property 0.0.8 exists to provide."""
        X, y, main = self._setup()
        W, R, S = self._reparam(np.random.default_rng(0))
        base = self._run(X, y, main, "roughness")
        moved = self._run(X, y, main, "roughness", W, R, S)
        assert moved[2] == base[2]                       # same column count
        assert abs(moved[0] - base[0]) < 1e-6            # evidence, in nats
        # edf carries the EM stopping tolerance rather than an exact identity.
        assert abs(moved[1] - base[1]) < 1e-4

    def test_ridge_was_not_invariant(self):
        """Characterizes what was replaced, so the contrast cannot rot."""
        X, y, main = self._setup()
        W, R, S = self._reparam(np.random.default_rng(0))
        base = self._run(X, y, main, "ridge")
        moved = self._run(X, y, main, "ridge", W, R, S)
        assert abs(moved[0] - base[0]) > 0.5

    def test_design_is_rank_deficient_but_penalty_null_space_is_dropped(self):
        """22 columns of design rank 16.

        Purification removes ``span[1, B_j, B_k]``, nine dimensions of the
        tensor's own column space. Three of those are the penalty's affine null
        space and are dropped outright -- keeping them would hand the zero
        function to ``kappa_j`` and break the invariance above. The remaining
        six are penalized but unidentified, contributing nothing to ``edf``
        because they lie in the null space of the design's Gram.
        """
        from srae.blocks import SplineBlock, TensorBlock

        X, y, main = self._setup()
        tb = TensorBlock(pair=(0, 1))
        Z = tb.fit(X[:, 0], X[:, 1], main)
        sv = np.linalg.svd(Z, compute_uv=False)
        assert Z.shape[1] == 22
        assert int(np.sum(sv > 1e-8 * sv[0])) == 16
        assert np.all(tb.s_ > 0)                 # no kappa direction remains

        from srae.inference import BlockSpec, fit_gaussian_eb
        blocks = [BlockSpec("t", slice(0, Z.shape[1]), tb.s_)]
        out = fit_gaussian_eb(Z, y, blocks, max_iter=800, tol=1e-11)
        assert out["edf"]["t"] < 16 + 1e-6       # bounded by the design rank

    def test_transform_round_trips_the_training_design(self):
        """``transform`` must reproduce ``fit`` on the training rows."""
        from srae.blocks import TensorBlock

        X, _, main = self._setup()
        tb = TensorBlock(pair=(0, 1))
        Z = tb.fit(X[:, 0], X[:, 1], main)
        assert np.allclose(tb.transform(X[:, 0], X[:, 1], main), Z, atol=1e-10)

    @pytest.mark.parametrize("scale", [1e-3, 1e-1, 1e1, 1e3])
    def test_gain_is_invariant_to_the_units_of_the_margins(self, scale):
        """Including *differential* scaling of the two margins.

        The three Kronecker terms carry different powers of the domain length,
        so an isotropic thin-plate penalty on raw knots would reweight them as
        a feature was re-expressed in other units. Rescaling each margin's
        knots to [0, 1] is what removes that.
        """
        from srae import SRAERegressor

        rng = np.random.default_rng(0)
        n = 400
        X = rng.normal(size=(n, 3))
        y = (2.0 * X[:, 0] * X[:, 1] + np.sin(1.5 * X[:, 2])
             + rng.normal(0, 0.3, n))

        def run(Xs):
            m = SRAERegressor(interactions="auto").fit(Xs, y)
            gain = [d["screen_gain"] for d in m.interactions_
                    if d["name"] == "x0*x1"]
            return m.evidence_, (gain[0] if gain else None)

        Xm = X.copy()
        Xm[:, 0] *= scale
        Xm[:, 1] *= scale / 3.0            # margins scaled differently
        base_ev, base_gain = run(X)
        moved_ev, moved_gain = run(Xm)

        assert base_gain is not None and moved_gain is not None
        assert moved_ev == pytest.approx(base_ev, rel=1e-9)
        assert moved_gain == pytest.approx(base_gain, rel=1e-6)

    def test_ridge_escape_hatch(self):
        """``penalty='ridge'`` restores the pre-0.0.8 construction."""
        from srae.blocks import TensorBlock

        class Legacy(TensorBlock):
            penalty = "ridge"

        X, _, main = self._setup()
        legacy = Legacy(pair=(0, 1))
        legacy.fit(X[:, 0], X[:, 1], main)
        assert np.allclose(legacy._penalty_matrix(5, 5), np.eye(25))
        assert TensorBlock.penalty == "roughness"

        current = TensorBlock(pair=(0, 1))
        current.fit(X[:, 0], X[:, 1], main)
        assert not np.allclose(current._penalty_matrix(5, 5), np.eye(25))


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
