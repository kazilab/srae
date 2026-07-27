"""scikit-learn estimator-protocol conformance for all four SRAE variants.

Split into three groups:

* :class:`TestEstimatorProtocol` -- clone / get_params / fitted metadata.
* :class:`TestIntegration`       -- Pipeline, cross_val_score, GridSearchCV.
* :class:`TestFeatureNameHandling` -- regression tests for two fixed bugs
  (see the class docstring); these are the ones most likely to silently
  reappear if the three ``fit`` implementations drift apart again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from conftest import COLUMNS, N_FEATURES, default_target, make, path_target

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# --------------------------------------------------------------------------
# Estimator protocol
# --------------------------------------------------------------------------

class TestEstimatorProtocol:
    def test_get_params_roundtrip(self, estimator_cls):
        m = make(estimator_cls)
        params = m.get_params()
        assert "n_knots" in params and "interactions" in params
        m.set_params(n_knots=7)
        assert m.get_params()["n_knots"] == 7

    def test_get_params_covers_init_signature(self, estimator_cls):
        """``**kwargs`` in ``__init__`` would silently drop params on clone."""
        import inspect

        sig = inspect.signature(estimator_cls.__init__)
        declared = {
            name for name, prm in sig.parameters.items()
            if name != "self" and prm.kind is not prm.VAR_KEYWORD
        }
        assert declared <= set(make(estimator_cls).get_params())

    def test_clone_preserves_params(self, estimator_cls):
        m = make(estimator_cls, n_knots=6)
        assert clone(m).get_params() == m.get_params()

    def test_clone_refit_is_reproducible(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls)
        first = m.fit(X, y).predict(X)
        second = clone(m).fit(X, y).predict(X)
        assert np.array_equal(first, second)

    def test_repr_does_not_fit(self, estimator_cls):
        assert estimator_cls.__name__ in repr(make(estimator_cls))

    def test_check_is_fitted(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        check_is_fitted(make(estimator_cls).fit(X, y))

    def test_fit_returns_self(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls)
        assert m.fit(X, y) is m


# --------------------------------------------------------------------------
# Fitted metadata -- all four variants, all three paths
# --------------------------------------------------------------------------

class TestFittedMetadata:
    @pytest.mark.parametrize("path", ["regression", "binary", "multiclass"])
    def test_n_features_in_(self, estimator_cls, path, X, targets):
        y = path_target(estimator_cls, targets, path)
        m = make(estimator_cls).fit(X, y)
        assert m.n_features_in_ == N_FEATURES

    @pytest.mark.parametrize("path", ["regression", "binary", "multiclass"])
    def test_feature_names_in_(self, estimator_cls, path, Xdf, targets):
        y = path_target(estimator_cls, targets, path)
        m = make(estimator_cls).fit(Xdf, y)
        assert list(m.feature_names_in_) == COLUMNS

    @pytest.mark.parametrize("path", ["regression", "binary", "multiclass"])
    def test_feature_names_fitted_attr(self, estimator_cls, path, Xdf, targets):
        y = path_target(estimator_cls, targets, path)
        m = make(estimator_cls).fit(Xdf, y)
        assert m.feature_names_ == COLUMNS

    def test_default_labels_without_columns(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls).fit(X, y)
        assert m.feature_names_ == [f"x{j}" for j in range(N_FEATURES)]

    def test_no_feature_names_in_for_plain_array(self, estimator_cls, X, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls).fit(X, y)
        assert not hasattr(m, "feature_names_in_")


# --------------------------------------------------------------------------
# Regression tests for two fixed bugs
# --------------------------------------------------------------------------

class TestFeatureNameHandling:
    """Guards against two bugs fixed in 0.0.6.

    1. ``fit`` wrote the resolved labels back onto the ``feature_names``
       constructor parameter.  Because the write was guarded on ``is None`` it
       latched the *first* dataset, so refitting on a differently-named frame
       produced a correct model carrying the previous dataset's labels.
    2. The multiclass ``fit`` overrides in ``pooled`` / ``scale_integration`` never
       called ``_set_sklearn_fit_attrs``, so multiclass Pooled/SI/SIPooled
       models had no ``n_features_in_`` or ``feature_names_in_`` at all.
    """

    @pytest.mark.parametrize("path", ["regression", "binary", "multiclass"])
    def test_fit_does_not_mutate_constructor_params(
        self, estimator_cls, path, Xdf, targets
    ):
        y = path_target(estimator_cls, targets, path)
        m = make(estimator_cls)
        before = m.get_params()
        m.fit(Xdf, y)
        assert m.get_params() == before

    @pytest.mark.parametrize("path", ["regression", "binary", "multiclass"])
    def test_refit_relabels_components(self, estimator_cls, path, X, targets):
        """Refitting on new column names must relabel, not keep stale ones."""
        y = path_target(estimator_cls, targets, path)
        first = pd.DataFrame(X, columns=COLUMNS)
        second_cols = ["red", "green", "blue", "violet"]
        second = pd.DataFrame(X, columns=second_cols)

        m = make(estimator_cls)
        m.fit(first, y)
        assert m.feature_names_ == COLUMNS
        m.fit(second, y)

        assert m.feature_names_ == second_cols
        assert list(m.feature_names_in_) == second_cols
        components = set()
        for name in m.summary()["component"]:
            components.update(str(name).split("*"))
        assert components <= set(second_cols)
        m.shape_function("red", **({"cls": m.classes_[0]} if path == "multiclass" else {}))

    def test_feature_names_in_cleared_on_array_refit(self, estimator_cls, X, Xdf, targets):
        y = default_target(estimator_cls, targets)
        m = make(estimator_cls).fit(Xdf, y)
        assert hasattr(m, "feature_names_in_")
        m.fit(X, y)
        assert not hasattr(m, "feature_names_in_")

    def test_explicit_names_win_over_columns(self, estimator_cls, Xdf, targets):
        y = default_target(estimator_cls, targets)
        explicit = ["p", "q", "r", "s"]
        m = make(estimator_cls, feature_names=explicit).fit(Xdf, y)
        assert m.feature_names_ == explicit
        assert m.get_params()["feature_names"] == explicit

    def test_multiclass_children_inherit_labels(self, classifier_cls, Xdf, y_mc):
        m = make(classifier_cls).fit(Xdf, y_mc)
        for est in m.estimators_:
            assert est.feature_names_ == COLUMNS


# --------------------------------------------------------------------------
# Integration with sklearn meta-estimators
# --------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.parametrize("path", ["regression", "binary", "multiclass"])
    def test_pipeline(self, estimator_cls, path, X, targets):
        y = path_target(estimator_cls, targets, path)
        pipe = Pipeline([("scale", StandardScaler()), ("srae", make(estimator_cls))])
        pipe.fit(X, y)
        assert pipe.predict(X).shape == (len(y),)

    @pytest.mark.parametrize("path", ["regression", "binary", "multiclass"])
    def test_cross_val_score(self, estimator_cls, path, X, targets):
        y = path_target(estimator_cls, targets, path)
        scores = cross_val_score(make(estimator_cls), X, y, cv=3)
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))

    @pytest.mark.parametrize("path", ["regression", "binary", "multiclass"])
    def test_grid_search(self, estimator_cls, path, X, targets):
        y = path_target(estimator_cls, targets, path)
        gs = GridSearchCV(make(estimator_cls), {"n_knots": [4, 5]}, cv=3)
        gs.fit(X, y)
        assert hasattr(gs, "best_estimator_")
        assert gs.predict(X).shape == (len(y),)

    def test_pipeline_with_dataframe(self, estimator_cls, Xdf, targets):
        y = default_target(estimator_cls, targets)
        pipe = Pipeline([("srae", make(estimator_cls))])
        pipe.fit(Xdf, y)
        assert pipe.named_steps["srae"].feature_names_ == COLUMNS


# --------------------------------------------------------------------------
# sklearn's own conformance suite (informational baseline)
# --------------------------------------------------------------------------

class TestCheckEstimator:
    """Pin the checks SRAE already satisfies so they cannot silently regress.

    SRAE deliberately does not implement sklearn's input-validation contract
    (sparse input, NaN/inf rejection, ``NotFittedError``), so a subset of
    ``check_estimator`` is expected to fail.  This test asserts only that the
    parameter-handling checks -- the ones the 0.0.6 fixes addressed -- pass.
    """

    REQUIRED = [
        "check_dont_overwrite_parameters",
        "check_estimators_overwrite_params",
        "check_get_params_invariance",
        "check_set_params",
        "check_estimators_dtypes",
        "check_fit_score_takes_y",
    ]

    def test_parameter_handling_checks_pass(self, estimator_cls):
        from sklearn.utils.estimator_checks import check_estimator

        results = check_estimator(make(estimator_cls, n_knots=4), on_fail=None)
        by_name = {}
        for r in results:
            by_name.setdefault(r["check_name"], set()).add(r["status"])

        failed = [
            name for name in self.REQUIRED
            if name in by_name and "failed" in by_name[name]
        ]
        assert not failed, f"{estimator_cls.__name__} regressed: {failed}"
