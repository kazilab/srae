"""Shared fixtures and the estimator registry used across the SRAE test suite.

Every test that can run against all four variants is parametrized over the
registries below, so a new variant only has to be added in one place.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from srae import (
    SRAEClassifier,
    SRAEClassifierSI,
    SRAEClassifierSIPooled,
    SRAEClassifierPooled,
    SRAERegressor,
    SRAERegressorSI,
    SRAERegressorSIPooled,
    SRAERegressorPooled,
)

# --------------------------------------------------------------------------
# Registries
# --------------------------------------------------------------------------

REGRESSORS = [
    SRAERegressor,
    SRAERegressorPooled,
    SRAERegressorSI,
    SRAERegressorSIPooled,
]

CLASSIFIERS = [
    SRAEClassifier,
    SRAEClassifierPooled,
    SRAEClassifierSI,
    SRAEClassifierSIPooled,
]

ALL_ESTIMATORS = REGRESSORS + CLASSIFIERS

# Variants running the hierarchical-pooling stack (edf budget, holdout
# calibration). Listed explicitly: the SI mixin also exposes ``pool_strength``,
# so an introspection-based check would wrongly include SRAE*SI.
POOLED = {
    SRAERegressorPooled,
    SRAEClassifierPooled,
    SRAERegressorSIPooled,
    SRAEClassifierSIPooled,
}

# Keep the suite fast: small designs, few knots, short MH chains.
_BASE_KW = dict(interactions=False, n_knots=5)
# Deliberately far below the shipped defaults, and single-chain: these settings
# do not converge and will trip the sampler's own warning, which is why the
# suite filters RuntimeWarning. Convergence behaviour is tested explicitly in
# TestScaleIntegrationDiagnostics rather than implied by every other test.
_SI_KW = dict(n_samples=6, n_burn=3, n_chains=1)

N_SAMPLES = 80
N_FEATURES = 4
COLUMNS = ["alpha", "beta", "gamma", "delta"]


def is_scale_integrated(cls) -> bool:
    """True for the MH-integrating variants (they accept ``n_samples``)."""
    return "n_samples" in inspect.signature(cls.__init__).parameters


def make(cls, **overrides):
    """Construct an estimator with fast, test-appropriate defaults."""
    kw = dict(_BASE_KW)
    if is_scale_integrated(cls):
        kw.update(_SI_KW)
    kw.update(overrides)
    return cls(**kw)


def is_classifier_cls(cls) -> bool:
    return cls in CLASSIFIERS


def is_pooled(cls) -> bool:
    """True for the hierarchical-pooling variants."""
    return cls in POOLED


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _latent(X):
    """Smooth additive signal with one genuine pairwise interaction."""
    return 1.2 * np.sin(1.5 * X[:, 0]) + 2.0 * X[:, 0] * X[:, 1] - 0.5 * X[:, 2]


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def X():
    return np.random.default_rng(0).normal(size=(N_SAMPLES, N_FEATURES))


@pytest.fixture(scope="session")
def Xdf(X):
    return pd.DataFrame(X, columns=COLUMNS)


@pytest.fixture(scope="session")
def y_reg(X):
    gen = np.random.default_rng(1)
    return _latent(X) + 0.3 * gen.normal(size=len(X))


@pytest.fixture(scope="session")
def y_bin(X):
    gen = np.random.default_rng(2)
    f = _latent(X)
    p = 1.0 / (1.0 + np.exp(-(f - f.mean())))
    return (gen.uniform(size=len(X)) < p).astype(int)


@pytest.fixture(scope="session")
def y_mc(X):
    """Three balanced classes."""
    f = _latent(X)
    return np.digitize(f, np.quantile(f, [1 / 3, 2 / 3]))


@pytest.fixture(scope="session")
def targets(y_reg, y_bin, y_mc):
    return {"regression": y_reg, "binary": y_bin, "multiclass": y_mc}


# Interaction screening on a Bernoulli likelihood needs appreciably more data
# than on a Gaussian one: at n=80 the planted term does not clear the default
# gain threshold, at n=400 it does.  Used only by the screening tests.
N_LARGE = 400


@pytest.fixture(scope="session")
def X_large():
    return np.random.default_rng(0).normal(size=(N_LARGE, N_FEATURES))


@pytest.fixture(scope="session")
def Xdf_large(X_large):
    return pd.DataFrame(X_large, columns=COLUMNS)


@pytest.fixture(scope="session")
def y_bin_large(X_large):
    gen = np.random.default_rng(2)
    f = _latent(X_large)
    p = 1.0 / (1.0 + np.exp(-(f - f.mean())))
    return (gen.uniform(size=len(X_large)) < p).astype(int)


def screening_data(cls, small, large):
    """Pick a dataset on which interaction screening can actually fire.

    ``small``/``large`` are ``(X, y)`` pairs for the regression and binary
    cases respectively.
    """
    return large if is_classifier_cls(cls) else small


def default_target(cls, targets):
    """The natural target for this estimator kind.

    Used by tests that exercise a shared surface (summary, plotting, cloning)
    and only need *some* valid target.
    """
    return targets["binary"] if is_classifier_cls(cls) else targets["regression"]


def path_target(cls, targets, path):
    """Target for an (estimator, path) pair, skipping impossible combinations.

    Regressors only run the ``regression`` path; classifiers only run
    ``binary`` / ``multiclass``.  Feeding a continuous target to a classifier
    would silently build one OvR model per distinct value.
    """
    if is_classifier_cls(cls):
        if path == "regression":
            pytest.skip("classifier: regression path not applicable")
        return targets[path]
    if path != "regression":
        pytest.skip("regressor: classification path not applicable")
    return targets["regression"]


# --------------------------------------------------------------------------
# Parametrization helpers
# --------------------------------------------------------------------------

def _ids(classes):
    return [c.__name__ for c in classes]


def pytest_generate_tests(metafunc):
    """Wire the registries into any test that asks for them by name."""
    if "regressor_cls" in metafunc.fixturenames:
        metafunc.parametrize("regressor_cls", REGRESSORS, ids=_ids(REGRESSORS))
    if "classifier_cls" in metafunc.fixturenames:
        metafunc.parametrize("classifier_cls", CLASSIFIERS, ids=_ids(CLASSIFIERS))
    if "estimator_cls" in metafunc.fixturenames:
        metafunc.parametrize("estimator_cls", ALL_ESTIMATORS, ids=_ids(ALL_ESTIMATORS))
    if "clf_path" in metafunc.fixturenames:
        metafunc.parametrize("clf_path", ["binary", "multiclass"])
