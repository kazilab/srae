# SRAE — Self-Regularizing Additive Estimator

SRAE is an evidence-screened empirical-Bayes additive model. It fits an
interpretable order-two functional-ANOVA model

```text
f(x) = intercept + sum_j f_j(x_j) + sum_(j,k in S) f_jk(x_j, x_k)
```

with penalized-spline main effects and a small set of automatically screened pairwise tensor interactions. Continuous roughness precisions, null-space shrinkage precisions, and (for Gaussian regression) residual variance are estimated from a marginal-likelihood objective. Gaussian regression uses exact posterior moments and EM updates conditional on the finite basis. Binary classification uses a Laplace approximation and posterior-moment fixed-point updates.

The package is best understood as an implementation-level synthesis of established GAM, penalized-spline, empirical-Bayes, and functional-ANOVA ideas. Its distinctive algorithmic element is the following workflow:

1. fit empirical-Bayes spline main effects;
2. score purified low-rank tensor candidates with a conditional residual marginal-likelihood calculation;
3. retain candidates above a fixed structural threshold; and
4. refit the selected main and interaction blocks jointly.

The residual score is not an exact Bayes factor for the complete model, especially for classification. Basis resolution, candidate-pair caps, evidence thresholds, and maximum interaction counts remain structural settings.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[test]"        # pytest suite
pip install -e ".[benchmark]"   # notebook and comparison dependencies
```

Requires Python 3.12+, numpy 2.5+, scipy 1.18+, pandas 3.0+, matplotlib 3.11+,
and scikit-learn 1.9+. The interpreter floor follows from the dependency floors
— numpy 2.5 and scipy 1.18 both require Python 3.12 — not from any language
feature SRAE uses.

## Minimal use

```python
from srae import SRAERegressor

model = SRAERegressor(
    n_knots=10,
    interactions="auto",
    interaction_gain_threshold=4.0,
    max_screen_pairs=40,
    max_interactions=8,
).fit(X_train, y_train)

prediction = model.predict(X_test)
lo, hi = model.predict_interval(X_test, level=0.90)
print(model.summary())
print(model.interactions_)
```

`SRAEClassifier` provides binary and multiclass classification. Binary
probabilities use posterior-variance moderation. Multiclass structure —
which blocks, which interaction pairs — is still discovered one-vs-rest, but
since 0.0.10 the selected structure is **refitted as a joint multinomial
model** with a Laplace posterior, and probabilities are moderated toward the
`K`-class neutral point `1/K`. That gives a coherent covariance between class
surfaces, which stacked binary fits cannot. Validate calibration on your own
data.

## Documentation

Full documentation — user guide with the model mathematics, an API reference, and a glossary — lives in `docs/` and builds with Sphinx:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

The project is configured for Read the Docs via `.readthedocs.yaml`. The build is warning-clean, so `fail_on_warning` is enabled.

Start with `docs/user_guide/` for the definitions behind each reported quantity:

- `model.rst` — functional-ANOVA decomposition, penalized-spline blocks, the exact `∫(f'')²` roughness penalty, penalty eigenbasis;
- `inference.rst` — the evidence objective, EM updates, Laplace approximation, effective degrees of freedom, and binary moderated probabilities;
- `interactions.rst` — tensor purification, the screening gain, the candidate pre-filter;
- `variants.rst` — the estimator grid and when each member is appropriate;
- `interpretation.rst` — how to read `summary()`, shape functions, and the evidence trace.

## sklearn compatibility

All public estimators (`SRAERegressor`, `SRAEClassifier`, and the pooled / scale-integrated variants) inherit from scikit-learn's `BaseEstimator` plus `RegressorMixin` / `ClassifierMixin`. They support the standard estimator protocol:

```python
from sklearn.base import clone, is_regressor, is_classifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from srae import SRAERegressor, SRAEClassifier

assert is_regressor(SRAERegressor())
assert is_classifier(SRAEClassifier())

# clone / get_params / set_params
est = clone(SRAERegressor(n_knots=8, interactions=False))

# cross-validation
scores = cross_val_score(
    SRAERegressor(n_knots=8, interactions=False), X, y, cv=5
)

# pipelines
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("srae", SRAERegressor(n_knots=8, interactions="auto")),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)

# hyperparameter search over structural settings
search = GridSearchCV(
    SRAEClassifier(interactions=False),
    param_grid={"n_knots": [6, 10], "max_iter": [50, 100]},
    cv=3,
)
search.fit(X_train, y_train)
```

Because SRAE estimates its shrinkage parameters internally, a grid search is only needed over *structural* settings such as `n_knots` or `interaction_gain_threshold` — not over penalty strengths.

Fitted estimators also expose `n_features_in_` and (when available) `feature_names_in_`.

Component labels used by `summary()`, `shape_function()`, and the plotting helpers are
resolved at fit time into `feature_names_` (explicit `feature_names` argument > DataFrame
columns > `x0…xp`). The `feature_names` *constructor parameter* is never written to by
`fit`, so refitting an estimator on a differently-named frame relabels it correctly.

## What the fitted object reports

- main-effect means and pointwise conditional empirical-Bayes intervals, plus
  fitted interaction mean surfaces;
- per-component effective degrees of freedom and learned precisions;
- selected pairwise interactions and their screening scores;
- an evidence or surrogate-evidence trace;
- Gaussian predictive intervals or Laplace-moderated binary probabilities.

These uncertainty summaries condition on estimated hyperparameters, the selected interaction set, and the fixed basis. They do not propagate interaction-selection uncertainty.

**Intervals are optimistic at small `n`.** On a well-specified Gaussian design (smooth additive truth, true `sigma^2 = 0.25`), nominal 90% predictive intervals covered, over 40 replicates:

| `n_train` | coverage | `sigma2_` (true 0.25) |
| --- | --- | --- |
| 60 | 86.3% | 0.225 |
| 100 | 88.5% | 0.238 |
| 200 | 89.2% | 0.245 |
| 400 | 89.7% | 0.250 |

Reproduce with `python benchmarks/run_coverage_sweep.py`, which uses the same design as `TestCoverage` in the test suite.

Coverage is essentially nominal by `n ≈ 200`. The shortfall is not a missing degrees-of-freedom correction — `sigma2_` already equals `RSS / (n - edf)` at the EM fixed point — but unpropagated *smoothing-parameter* uncertainty, a known limitation of empirical-Bayes GAM intervals. Integrating that scale recovers most of it: `SRAERegressorSI` reached 89.9% at `n = 100` against 88.5% for the Type-II estimator. Prefer the scale-integrated variants when calibration matters at small `n`.

Because the truth here is smooth, additive and well specified, these are best-case figures — an upper bound on what to expect from correlated or misspecified real data.

## Estimator variants

Eight estimators span a 2x2 grid over two independent axes, crossed with the
task. They share the same model family, block constructors, and screening
algorithm and settings. Their main difference is how the block
hyperparameters are treated. The selected interaction set is not guaranteed
to match across variants, because each variant's main-effect fit determines
the residuals passed to screening.

| Regularization | Hyperprior | Regression | Classification |
| --- | --- | --- | --- |
| Type-II MLE | point estimate | `SRAERegressor` | `SRAEClassifier` |
| pooled stack | point estimate | `SRAERegressorPooled` | `SRAEClassifierPooled` |
| Type-II MLE | integrated | `SRAERegressorSI` | `SRAEClassifierSI` |
| pooled stack | integrated | `SRAERegressorSIPooled` | `SRAEClassifierSIPooled` |

The pooled and global-scale integration classes were developed for small-sample experiments. They are not part of the core preprint claims and their behaviour, priors, and calibration should be validated separately before scientific or regulated use.

Two properties are easy to get wrong and are worth stating here:

- **`evidence_` is not comparable across variants.** On a fixed design, the pooled stack moves away from the unpooled evidence optimum. Each scale-integrated response reports a *mean log evidence* over a posterior whose prior is truncated by default to scale factors at least as regularized as the MAP; a multiclass parent sums those per-head means. Screening can also produce different fitted designs. An evidence ranking is therefore not meaningful and will often favour the plain Type-II estimator, but no universal ordering is guaranteed. Compare by held-out score only.
- **The pooled capacity cap is aggressive.** On every dataset in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) the pooled stack scored below the plain Type-II estimator. Verify against the plain estimator rather than assuming the pooled variant is uniformly safer; it was developed for small-sample settings those datasets do not represent.
- **SI memory is bounded.** Scale factors are stored for every retained draw (so ESS / R-hat use the full chain), but full `beta` / `Sigma` matrices are kept only for a thinned subsample of at most 128 draws. That avoids multi-GB storage when the design is large; the predictive average then uses that finite subsample (higher Monte Carlo variance than using every draw). See `_MAX_STORED_DRAWS` in `scale_integration.py`.

See `docs/user_guide/variants.rst` for the mathematics of each axis.

Every measurement quoted in this project is reproduced by a committed script in
`benchmarks/`: [`RESULTS.md`](benchmarks/RESULTS.md) from
`run_benchmarks.py` (four public datasets, five-fold cross-validation, fixed
seed, regenerated for 0.0.10), and the coverage table above from
`run_coverage_sweep.py`. Nothing is cited that cannot be regenerated. TabPFN is
absent from the current benchmark run: it needs a gated download and local
weights, so the script skips it and the header records that.

## Important limitations

- The model is restricted to smooth main effects and selected pairwise interactions.
- Interaction discovery is greedy and conditional on the main-effect fit.
- Product-correlation pre-ranking can miss interactions that are symmetric, masked, or poorly represented by a centered product.
- `interaction_gain_threshold` remains tied to the structural settings — marginal resolution, purification, and the isotropic combination of second derivatives — even though since 0.0.8 it no longer depends on how the marginals are parametrized.
- Conditional residual screening leaks: when a strong interaction is present, pairs sharing a feature with it score above the threshold too. The genuine pair ranks far higher, but `max_interactions` is what bounds the consequence.
- Interaction screening has markedly less power under a Bernoulli likelihood than a Gaussian one. On a design where the planted pair is recovered at n=80 for regression, classification did not select it until roughly n=400. An empty `interactions_` on a small classification problem indicates lack of power, not absence of structure.
- The logistic procedure is approximate; evidence monotonicity is not guaranteed by the Gaussian EM argument.
- Multiclass *structure* — which blocks, which interaction pairs — is still
  discovered with independent one-vs-rest Bernoulli fits; only the refit is
  joint. Screening is therefore uncoupled across classes.
- The joint refit is declined above a `(K-1) × n_columns` limit of 4000, with a
  warning, because its Hessian is that square. Such fits fall back to the
  `normalized_ovr` link, as do the pooled and scale-integrated variants, which
  keep their own multiclass paths.
- The current implementation uses dense linear algebra and rejects neither sparse input nor NaN/inf explicitly — such input fails inside numpy rather than with a scikit-learn-standard error.
- Input imputation, encoding, and leakage-safe preprocessing are external responsibilities.

## Repository layout

```text
srae/                          core estimators, inference engines, plotting
  blocks.py                    spline / linear / factor / purified tensor blocks,
                               exact roughness penalties
  inference.py                 Gaussian, logistic and joint multinomial engines
  model.py                     SRAERegressor, SRAEClassifier
  pooled.py                    pooled anti-overfitting variants
  scale_integration.py         hyperparameter-scale integration variants
  plotting.py                  shape, interaction, importance, evidence plots
tests/                         pytest suite (API, sklearn conformance, statistical checks)
  conftest.py                  estimator registry; every variant-parametrized test
  test_api.py                  modelling API, penalties, interactions, multiclass
  test_sklearn_compat.py       estimator protocol / check_estimator pins
  test_statistical.py          coverage, null behaviour, robustness (synthetic)
  test_metadata.py             version / license single-source-of-truth checks
benchmarks/                    every published measurement is reproduced here
  run_benchmarks.py            public datasets vs standard baselines
  RESULTS.md                   its recorded output, regenerate rather than edit
  run_coverage_sweep.py        predictive-interval coverage against sample size
docs/                          Sphinx documentation sources
CHANGELOG.md                   release history and behaviour changes
```

## Changelog

Release history, including behaviour changes that alter fitted results, is in
[`CHANGELOG.md`](CHANGELOG.md). Read it before upgrading. Two recent defaults
changed fitted results: 0.0.10 made multiclass estimation a joint multinomial
Laplace fit, which changes every multiclass probability and `evidence_`;
0.0.8 replaced the tensor interaction penalty with the
surface roughness, making the screening gain independent of the marginal basis;
0.0.7 replaced the spline roughness penalty with the exact `∫(f'')²`. Both move
every reported `evidence_`, `edf_`, and `lam_`. 0.0.6 changed the multiclass
probability coupling to a softmax. 0.0.9 changes no fitted quantity, but
spline `n_coef` drops by one and `kappa` rescales.

## Testing

```bash
pytest
```

The suite parametrizes over the estimator variants across regression, binary classification, and multiclass. It covers the modelling API, reporting surface, interaction discovery, plotting, scikit-learn protocol conformance, and a synthetic statistical layer (`test_statistical.py`: interval coverage trends, null interaction rates, collinearity / heteroskedasticity / OOD clamp behaviour).

Alongside those, one group of tests exists specifically to pin *mathematical* properties that would otherwise regress silently — each was written after a real defect was found:

| Test class | Pins |
| --- | --- |
| `TestRoughnessPenalty` | the spline penalty equals `∫(f'')²` against quadrature; its null space is exactly the straight lines; fits are invariant to the units of `x` |
| `TestTensorPenalty` | the tensor penalty equals its double integral; the screening gain is invariant to the marginal basis, which the pre-0.0.8 ridge was not |
| `TestJointMultinomial` | analytic gradient and Hessian against finite differences; exact reduction to the logistic engine at `K = 2`; invariance to class relabelling |

Thresholds in the statistical tests are intentionally loose: they encode defensible properties, not paper-grade benchmarks. Real datasets, distribution shift, and formal FDR calibration remain external validation responsibilities.

## Verification status

- `pytest`: 559 tests pass (96 skipped) across the supported interpreter range, Python 3.12 and 3.13, against the declared dependency floors (numpy 2.5, scipy 1.18, pandas 3.0, matplotlib 3.11, scikit-learn 1.9). Both versions are audited directly, and the suite is also run from the built sdist; `.github/workflows/ci.yml` runs the whole matrix on every push and pull request. Skips are deliberate regressor-x-classification and classifier-x-regression exclusions.
- `sphinx-build -W`: documentation builds with zero warnings; 24 documentation doctests pass.
- `sklearn.utils.estimator_checks.check_estimator`: the parameter-handling checks — including `check_dont_overwrite_parameters` and `check_estimators_overwrite_params` — pass across all eight public estimators and are pinned by regression tests. Exact totals are intentionally not stated because scikit-learn changes its check inventory between releases.
- Remaining `check_estimator` failures are absent input validation: sparse input, NaN/inf rejection, `NotFittedError` versus `RuntimeError`, and 1-D/2-D `y` handling. These are known gaps, not silent misbehaviour.

MIT license. For the installed version, see `srae.__version__`.
