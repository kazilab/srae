"""Statistical behaviour, as distinct from API behaviour.

The rest of the suite checks that the estimators *run* and honour the sklearn
protocol.  These tests check that what they report is statistically defensible:
interval coverage, false discovery under a null, and stability under
correlated predictors and heteroskedasticity.

Thresholds are deliberately loose.  Each test asserts a property that should
hold for any correct implementation, not the specific numbers this version
happens to produce -- a tight bound here would fail on a harmless change and
teach the next person to delete the test.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from srae import SRAEClassifier, SRAERegressor, SRAERegressorSI

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _additive(X):
    return 1.2 * np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] - 0.5 * X[:, 2]


def _sample(seed, n, sigma=0.5, p=3, f=_additive):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    return X, f(X) + rng.normal(0, sigma, n)


# --------------------------------------------------------------------------
# Predictive interval coverage
# --------------------------------------------------------------------------

class TestCoverage:
    """Nominal 90% intervals must be in the right neighbourhood.

    Coverage is known to be optimistic at small n because smoothing-parameter
    uncertainty is not propagated (see ``SRAERegressor.predict_interval``).
    These bounds encode "usable, and converging to nominal", not "exact".
    """

    @staticmethod
    def _coverage(cls, n, reps=12, level=0.9, sigma=0.5, **kw):
        z = norm.ppf(0.5 + level / 2)
        out = []
        for r in range(reps):
            X, y = _sample(500 + r, n, sigma)
            Xt, yt = _sample(9000 + r, 400, sigma)
            m = cls(interactions=False, **kw).fit(X, y)
            mean, std = m.predict(Xt, return_std=True)
            out.append(np.mean(np.abs(yt - mean) <= z * std))
        return float(np.mean(out))

    def test_large_n_coverage_is_near_nominal(self):
        cov = self._coverage(SRAERegressor, n=400)
        assert 0.86 < cov < 0.94

    def test_coverage_improves_with_n(self):
        """The shortfall is an n-effect, so it must shrink as n grows."""
        small = self._coverage(SRAERegressor, n=60)
        large = self._coverage(SRAERegressor, n=400)
        assert large > small

    def test_scale_integration_is_not_worse_at_small_n(self):
        """Integrating the hyperparameter scale should not lose coverage."""
        plain = self._coverage(SRAERegressor, n=100, reps=8)
        integ = self._coverage(SRAERegressorSI, n=100, reps=8,
                               n_samples=300, n_burn=200, n_chains=2)
        assert integ >= plain - 0.01

    def test_intervals_widen_with_noise(self):
        X, y = _sample(0, 300, sigma=0.2)
        Xn, yn = _sample(0, 300, sigma=1.0)
        quiet = SRAERegressor(interactions=False).fit(X, y)
        loud = SRAERegressor(interactions=False).fit(Xn, yn)
        lo_q, hi_q = quiet.predict_interval(X, level=0.9)
        lo_l, hi_l = loud.predict_interval(Xn, level=0.9)
        assert np.mean(hi_l - lo_l) > np.mean(hi_q - lo_q)


# --------------------------------------------------------------------------
# False discovery under a null
# --------------------------------------------------------------------------

class TestNullBehaviour:
    """Pure noise must not produce confident structure."""

    def test_no_interactions_found_under_pure_noise(self):
        """y independent of X: the screening gain threshold should hold."""
        found = 0
        reps = 15
        for r in range(reps):
            rng = np.random.default_rng(2000 + r)
            X = rng.normal(size=(300, 4))
            y = rng.normal(size=300)          # no relationship at all
            m = SRAERegressor(interactions="auto").fit(X, y)
            found += len(m.interactions_)
        # A screening rule that fires often on pure noise is not usable.
        assert found / reps < 0.5

    def test_additive_truth_does_not_manufacture_interactions(self):
        """A purely additive signal must not be explained by a pair."""
        found = 0
        reps = 10
        for r in range(reps):
            X, y = _sample(3000 + r, 400, sigma=0.4)
            m = SRAERegressor(interactions="auto").fit(X, y)
            found += len(m.interactions_)
        assert found / reps < 1.0

    @staticmethod
    def _orthogonal_null(seed, n=400):
        """x1 exactly orthogonal to the centered response: beta is exactly 0."""
        rng = np.random.default_rng(seed)
        x0 = rng.normal(size=n)
        y = 1.5 * x0 + rng.normal(0, 0.3, n)
        y -= y.mean()
        x1 = rng.normal(size=n)
        x1 -= x1.mean()
        x1 = x1 - y * (y @ x1) / (y @ y)
        return np.column_stack([x0, x1]), y

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_null_feature_is_listed_in_at_boundary(self, seed):
        """The flag must fire at *default* settings, not just long runs.

        Asserting membership in ``at_boundary_`` rather than comparing edfs:
        an edf comparison passes for any feature weaker than the signal and
        so would not detect a mis-calibrated threshold.
        """
        X, y = self._orthogonal_null(seed)
        m = SRAERegressor(interactions=False,
                          feature_types=["auto", "linear"]).fit(X, y)
        assert "x1" in m.at_boundary_
        assert "x0" not in m.at_boundary_

    def test_weak_but_real_signal_is_not_flagged(self):
        """The threshold must not label a genuine weak effect as pruned."""
        rng = np.random.default_rng(4)
        n = 400
        X = rng.normal(size=(n, 2))
        y = 1.5 * X[:, 0] + 0.05 * X[:, 1] + rng.normal(0, 0.3, n)
        m = SRAERegressor(interactions=False,
                          feature_types=["auto", "linear"]).fit(X, y)
        assert m.at_boundary_ == []

    def test_null_classifier_stays_near_base_rate(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(300, 3))
        y = rng.integers(0, 2, 300)
        m = SRAEClassifier(interactions=False).fit(X, y)
        p = m.predict_proba(X)[:, 1]
        # Predictions may drift, but not to confident extremes on noise.
        assert p.min() > 0.02 and p.max() < 0.98


# --------------------------------------------------------------------------
# Robustness of the reported fit
# --------------------------------------------------------------------------

class TestRobustness:
    def test_correlated_predictors_do_not_break_the_fit(self):
        """Near-collinear inputs: predictions must stay sane and finite."""
        rng = np.random.default_rng(5)
        n = 300
        x0 = rng.normal(size=n)
        x1 = x0 + 0.01 * rng.normal(size=n)        # correlation ~0.9999
        X = np.column_stack([x0, x1, rng.normal(size=n)])
        y = 1.5 * x0 + rng.normal(0, 0.3, n)
        m = SRAERegressor(interactions=False).fit(X, y)
        pred = m.predict(X)
        assert np.all(np.isfinite(pred))
        assert m.score(X, y) > 0.5
        # Total attributed capacity must not blow up on a duplicated signal.
        assert sum(m.edf_.values()) < n / 2

    def test_duplicated_feature_splits_rather_than_doubles(self):
        """An exact duplicate must not double the fitted effect."""
        rng = np.random.default_rng(6)
        n = 300
        x0 = rng.normal(size=n)
        y = 1.5 * x0 + rng.normal(0, 0.3, n)
        single = SRAERegressor(interactions=False).fit(x0.reshape(-1, 1), y)
        dup = SRAERegressor(interactions=False).fit(
            np.column_stack([x0, x0]), y)
        assert np.allclose(single.predict(x0.reshape(-1, 1)),
                           dup.predict(np.column_stack([x0, x0])), atol=0.15)

    def test_heteroskedastic_noise_is_absorbed_not_amplified(self):
        """Variance is assumed constant; the fit must degrade gracefully."""
        rng = np.random.default_rng(8)
        n = 400
        X = rng.normal(size=(n, 3))
        scale = 0.2 + 0.8 * np.abs(X[:, 0])        # noise grows with x0
        y = _additive(X) + rng.normal(0, 1, n) * scale
        m = SRAERegressor(interactions=False).fit(X, y)
        assert np.all(np.isfinite(m.predict(X)))
        assert m.sigma2_ > 0
        # sigma2_ should land between the extremes of the true variance.
        assert scale.min() ** 2 < m.sigma2_ < scale.max() ** 2

    def test_shifted_inputs_extrapolate_without_exploding(self):
        """Outside the training range the basis clamps; predictions stay finite."""
        X, y = _sample(13, 300)
        m = SRAERegressor(interactions=False).fit(X, y)
        far = np.full((20, 3), 12.0)               # far outside training support
        pred = m.predict(far)
        assert np.all(np.isfinite(pred))
        assert np.abs(pred).max() < 50 * np.abs(y).max()

    def test_constant_column_is_handled(self):
        X, y = _sample(15, 200)
        X[:, 2] = 3.0                              # zero-variance feature
        m = SRAERegressor(interactions=False).fit(X, y)
        assert np.all(np.isfinite(m.predict(X)))


# --------------------------------------------------------------------------
# Invariance to nominal category coding
# --------------------------------------------------------------------------

class TestFactorCodingInvariance:
    """Relabelling a nominal factor must not change the fit.

    ``feature_types='factor'`` fixed the main effect, but the tensor marginal
    kept splining the raw category codes, so permuting the labels of an
    eight-level factor moved test R^2 from 0.34 to 0.10.
    """

    @staticmethod
    def _factor_interaction(n=1200, seed=0):
        rng = np.random.default_rng(seed)
        lev = rng.integers(0, 8, n)
        z = rng.normal(size=n)
        effect = np.array([2.0, -1.5, 0.5, -2.0, 1.0, 1.5, -0.5, -1.0])
        y = effect[lev] * z + 0.3 * z + rng.normal(0, 0.4, n)
        return lev, z, y

    def _fit_pred(self, perm, lev, z, y, n_tr=800):
        X = np.column_stack([perm[lev].astype(float), z])
        m = SRAERegressor(feature_types=["factor", "auto"],
                          interactions="auto").fit(X[:n_tr], y[:n_tr])
        return m.predict(X[n_tr:]), m

    def test_relabelling_does_not_change_predictions(self):
        lev, z, y = self._factor_interaction()
        perm = np.array([3, 7, 0, 5, 1, 6, 2, 4])
        p_id, m_id = self._fit_pred(np.arange(8), lev, z, y)
        p_pm, m_pm = self._fit_pred(perm, lev, z, y)
        rmse = float(np.sqrt(np.mean((p_id - p_pm) ** 2)))
        # Was ~1.0 (vs an outcome sd of ~1.35) before factor-aware marginals.
        assert rmse < 0.1 * y.std()
        assert ([d["name"] for d in m_id.interactions_]
                == [d["name"] for d in m_pm.interactions_])

    def test_factor_interaction_is_actually_recovered(self):
        """A dummy-tensor basis should fit a factor x continuous interaction."""
        lev, z, y = self._factor_interaction()
        X = np.column_stack([lev.astype(float), z])
        m = SRAERegressor(feature_types=["factor", "auto"],
                          interactions="auto").fit(X[:800], y[:800])
        from sklearn.metrics import r2_score
        assert r2_score(y[800:], m.predict(X[800:])) > 0.7

    def test_tensor_marginal_is_indicator_for_factors(self):
        from srae.blocks import TensorBlock

        rng = np.random.default_rng(0)
        lev = rng.integers(0, 4, 200).astype(float)
        tb = TensorBlock((0, 1), levels=(np.arange(4.0), None))
        tb._ts = []
        B = tb._marginal(lev, fit=True, side=0)
        assert B.shape == (200, 4)
        assert set(np.unique(B)) <= {0.0, 1.0}
        assert np.all(B.sum(axis=1) == 1)
