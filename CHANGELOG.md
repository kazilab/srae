# Changelog

All notable changes to SRAE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). While the major
version is 0, behaviour changes that alter fitted results may appear in a patch
release; each one is listed under **Changed** with the reason and the escape
hatch, if there is one.

The single source of truth for the version is `srae/_metadata.py`.

## [0.0.10] — 2026-08-01

### Added

- **A joint multinomial Laplace engine, `fit_multinomial_eb`**, and with it a
  genuine joint posterior for multiclass models. This is the fix the
  `_multiclass_proba` docstring had been calling "the principled fix" since
  0.0.6 while shipping something else.

  Independent one-vs-rest fits leave the cross-class blocks of the Hessian at
  zero, so they carry no covariance between class surfaces and no neutral point
  to moderate toward. The engine fits `K-1` contrasts on a shared design under
  one softmax likelihood, with Hessian blocks
  `Z' diag(P C_a * C_b - (PC_a)(PC_b)) Z`.

  Validated against: finite-difference gradient and Hessian, a general-purpose
  optimizer for the MAP, and — the strongest check — **exact reduction to the
  logistic engine at K = 2** (evidence and edf agree to 1e-8; coefficients
  agree after the fixed `sqrt(2)` contrast scaling).

### Changed

- **`multiclass_link` now defaults to `"joint"`.** Multiclass probabilities
  change. The logits are moderated toward the `K`-class neutral point,
  `p = softmax(eta / sqrt(1 + pi * vbar / 8))`, with `vbar` the mean variance
  of the pairwise logit contrasts. A *common* factor per row keeps the link
  independent of class labelling, and at `K = 2` it reduces exactly to the
  binary moderated probability.

  Measured held-out on synthetic multiclass data, 5 seeds, 5000 test points:

  | K | n | link | log-loss | ECE | accuracy |
  |---|---|---|---|---|---|
  | 3 | 1500 | **joint** | **0.9413** | **0.0191** | **0.5520** |
  | 3 | 1500 | softmax | 0.9762 | 0.0942 | 0.5499 |
  | 3 | 1500 | normalized_ovr | 0.9437 | 0.0267 | 0.5498 |
  | 10 | 300 | **joint** | **2.0900** | **0.0132** | **0.2244** |
  | 10 | 300 | softmax | 2.1955 | 0.0837 | 0.2090 |
  | 10 | 300 | normalized_ovr | 2.1451 | 0.0353 | 0.2083 |

  The joint route wins log-loss in every configuration tried (K in 3/5/10,
  n in 300/1500), is best or tied on ECE, and improves accuracy — by 1.5
  points at K = 10.

- **The 0.0.6 softmax change was a regression, and this corrects it.** That
  release replaced `normalized_ovr` on a theoretical argument, without a
  calibration measurement. Measured now, plain softmax over one-vs-rest logits
  is the *worst* of the three routes: 2-4x the ECE of the route it replaced,
  higher log-loss, on 5/5 seeds at every K and n. Coherent row sums are not
  calibrated probabilities.

  The theoretical argument was also partly wrong. It objected that per-head
  moderation shrinks toward 0.5 rather than `1/K` — but dividing by the row sum
  maps a uniform shrinkage toward 0.5 exactly onto a shrinkage toward `1/K`.
  Decomposing moderation x coupling shows the *coupling* drives the difference
  and the moderation barely registers. The genuine defect in `normalized_ovr`
  is incoherent row sums, measured between 0.37 and 1.83.

- **Class logits use a sum-to-zero contrast basis, not a reference class.**
  Caught by the permutation test rather than by inspection: a reference-class
  parametrization shrinks every logit toward whichever class is the reference,
  and relabelling the classes moved fitted probabilities by up to **0.68**.
  Parametrizing as `beta = C gamma` with `C'C = I`, `C'1 = 0` puts an isotropic
  prior on a permutation-invariant subspace, so the fit is now invariant to
  class labelling (pinned to 1e-6 by refitting on cyclically relabelled
  classes).

- **`evidence_` for multiclass is now the joint marginal likelihood** rather
  than the sum of the one-vs-rest evidences. The old value was not a marginal
  likelihood at all: summing `K` binary evidences scores the same data `K`
  times.

- Block precisions are warm-started from the one-vs-rest children (geometric
  mean across classes), and the EM step computes only `diag(Sigma)` from the
  Cholesky factor rather than a full inverse. Without both, the joint stage
  dominated fitting time.

### Kept as-is, deliberately

- **`summary()`, `shape_function()` and the plotting helpers still report the
  one-vs-rest children.** The joint model is a *contrast* parametrization with
  no coefficients for any single class in isolation, whereas the children carry
  an absolute surface per class — which is what those views are for. The joint
  fit is exposed separately as `joint_`, and `beta_` / `Sigma_` / `edf_` on the
  parent remain what they always were: binary-only.
- **Interaction screening remains one-vs-rest.** Structure discovery is
  orthogonal to the calibration problem and was already documented as
  uncoupled; the joint stage refits the union of what screening selected.
- **Pooled and scale-integrated variants keep their own multiclass paths.**
  They override the multiclass fit with machinery (edf budget, MH sampling over
  scales) that has no joint analogue yet. Their fallback is now
  `normalized_ovr` rather than `softmax`, since softmax measured worse on both
  metrics.

- The joint refit is declined above `_MAX_JOINT_DIM = 4000` for
  `(K-1) * n_columns`, with a `RuntimeWarning` naming the dimension and the
  fallback. The Hessian is that square, so memory grows as its square and time
  as its cube: sklearn's own `check_estimator` fits a 200-class problem that
  would ask for a 14129-square Hessian — 1.6 GB per matrix, and enough of them
  to be OOM-killed. The limit admits 10 classes over 40 features (3249, about
  half a minute) and declines rather than degrading silently.

### Documentation and test suite

- Swept every surface still describing pre-0.0.10 multiclass behaviour: the
  `multiclass_link` and `predict_proba` docstrings, `evidence_` (documented as
  a sum over sub-models, now the joint marginal likelihood), the glossary's
  `one-vs-rest` and moderated-probability entries, `docs/index.rst`,
  `interpretation.rst`, and the README's limitations. Added a
  `multiclass_link` section to `inference.rst` with the moderation formula and
  the measured comparison. Documented `joint_` as an attribute.
- **`benchmarks/RESULTS.md` regenerated for 0.0.10**, clearing four releases of
  staleness. The penalty and multiclass work shows up on real data:

  | dataset | metric | 0.0.5 | 0.0.10 |
  |---|---|---|---|
  | california(sub) | R² | 0.7324 | **0.7661** |
  | california(sub) | train−test gap | 0.0851 | **0.0488** |
  | wine (3-class) | log-loss | 0.2013 | **0.1666** |
  | breast_cancer | accuracy | 0.9649 | **0.9754** |
  | diabetes | R² | 0.4959 | **0.5001** |

  California has strongly skewed features, which is exactly where 0.0.7
  predicted the exact roughness penalty would help — and it does, with the
  generalization gap nearly halving. Wine is the multiclass case: log-loss
  improves 17% at unchanged accuracy, the real-data counterpart of the
  synthetic calibration result behind 0.0.10. Interval coverage stays nominal
  throughout (89.8-91.2% against a 90% target).

  TabPFN is absent from the new run — it needs a gated download and local
  weights, so the script skips it and the header records `not installed`. EBM
  is present via `interpret-core`, avoiding the `shap`/`llvmlite` chain that
  cannot build on Python 3.13.
- README: the Testing section now names what the property-based test classes
  pin, and the repository layout lists `conftest.py`, `test_metadata.py` and
  `benchmarks/`.
- pytest config: `-ra` so the 96 deliberate skips report their reasons rather
  than appearing as a bare count, plus `--strict-markers`, `--strict-config`
  and `xfail_strict`.

### Known issues

- Multiclass fitting is slower: the joint stage adds roughly 1.2-5x the
  one-vs-rest cost, scaling with the Hessian dimension `(K-1)p`. That is
  inherent to fitting one `(K-1)p`-dimensional Laplace posterior instead of `K`
  independent `p`-dimensional ones, not an implementation artifact.
- The calibration figures behind the multiclass change are synthetic, single
  DGP family, 5 seeds. `benchmarks/RESULTS.md` now corroborates the direction
  on one real 3-class dataset, but broader real-data validation remains an
  external responsibility — as the README has always said.
- **Every published measurement is now reproduced by a committed script.**
  Two sets of figures were quoted from experiments whose code was never
  committed, so neither could be checked or regenerated:

  - The nine-regime pooled/SI sweep in `variants.rst` (`n` 50–400, `p` 8–60),
    including the "R² 0.99 vs 0.53" claim echoed in the README and the
    `SRAERegressorPooled` docstring. **Removed.** The qualitative points it
    supported are true and are now stated from `RESULTS.md` instead, which
    does show the pooled stack behind the plain estimator on all four
    datasets, and SI improving held-out log-loss on both classification ones.
  - The README's interval-coverage table. **Kept, but regenerated and made
    reproducible** by a new `benchmarks/run_coverage_sweep.py`, whose design
    matches `TestCoverage` in the test suite so the documented numbers and the
    asserted bounds describe the same experiment. Over 40 replicates coverage
    is 86.3 / 88.5 / 89.2 / 89.7% at `n` = 60 / 100 / 200 / 400 (previously
    85.1 / 87.4 / 89.3 / 89.8), and scale integration lifts `n` = 100 from
    88.5% to 89.9%.

## [0.0.9] — 2026-07-31

### Changed

- **Spline blocks now carry one zero-penalty column instead of two collinear
  ones.** `n_coef` for a default spline block drops from 14 to 13, and the
  reported `kappa` rescales — by exactly 2 when the two columns were equally
  scaled.

  The order-2 roughness penalty annihilates a two-dimensional space, but
  centering removes the constant *fitted* contribution, so exactly one function
  survives: a straight line in raw `x`. An eigensolver returns an arbitrary
  mixture of the two zero-eigenvalue vectors — in practice always one leaving
  both columns non-constant, measured as 2 columns on every one of four
  x-distributions × six seeds — so the norm filter kept two perfectly collinear
  columns for that single function.

  **No fitted quantity changes.** Two collinear coordinates sharing an
  isotropic `κ_j` telescope to the same EM fixed point as one. Measured against
  the redundant parametrization built by hand: evidence, edf, `λ`, and fitted
  values agree to EM stopping tolerance (~1e-3 nats against a 4.0-nat selection
  threshold). What the redundancy cost was reporting — `n_coef` overstated by
  one, and the documented bound "edf cannot exceed the block's column count"
  loose by one for no reason.

  The retained column is exactly a straight line in raw `x` (verified to
  ~1e-12 across four x-distributions), completing what 0.0.7 started: the
  `κ_j` direction is now both exactly linear *and* exactly one coefficient.

  `LinearBlock` and `FactorBlock` are untouched — neither rotates into a
  penalty eigenbasis. `TensorBlock` already handled its own null space
  canonically as of 0.0.8, by dropping it; the two block types are now
  consistent in treating the penalty null space deliberately rather than
  leaving it to an eigensolver's arbitrary mixture.

### Added

- `TestRoughnessPenalty.test_null_space_is_one_canonical_column`, parametrized
  over four x-distributions, asserts one zero-penalty column and that no two
  design columns are collinear.
- `TestRoughnessPenalty.test_canonicalizing_the_null_space_changes_no_fitted_quantity`
  rebuilds the redundant parametrization by hand and pins that evidence, edf,
  `λ`, and fitted values are unchanged — so the collapse cannot silently become
  a modelling change.

## [0.0.8] — 2026-07-31

### Changed

- **The tensor interaction penalty is now the roughness of the fitted surface**,
  `∫∫[f_jj² + 2f_jk² + f_kk²]`, replacing the isotropic ridge on tensor
  B-spline coefficients. **Interaction screening decisions and every reported
  `evidence_`/`edf_` for tensor blocks change.**

  A ridge is not a functional of the surface — `I` is the identity in whatever
  coordinates it is written — so reparametrizing the marginals over the *same*
  function space moved the screening gain. The roughness penalty transforms
  correctly (`Ω → WᵀΩW` alongside `T_p → T_p W`), leaving the induced prior
  `T_p Ω⁺ T_pᵀ` unchanged. Measured on a continuous × continuous pair:

  | penalty | basis dependence of the gain | edf |
  |---|---|---|
  | isotropic ridge (to 0.0.7) | 1.7 nats | 11.6 |
  | roughness (0.0.8) | 8e-11 nats | 8.3 |

  The roughness penalty is also better supported on that pair (~12 nats) while
  spending less capacity. Computed in Kronecker form from one-dimensional
  derivative and Gram matrices, with each margin's knots rescaled to `[0,1]` —
  the three terms carry different powers of the domain length, so on raw knots
  their relative weighting would drift with the units of a feature.

  A single `λ_jk` per surface is kept rather than one per margin: the
  anisotropic form would double the hyperparameters of every candidate pair
  during screening, for a surface that is discarded unless it clears the gain
  threshold.

  Nominal sides have no smoothness to penalize, so all three one-dimensional
  parts become the identity. The penalty then reduces to a Sobolev-type
  penalty applied independently per category, which is permutation-invariant
  in the levels and so preserves the coding invariance the indicator marginals
  exist to provide.

  Set the `TensorBlock.penalty` **class** attribute to `"ridge"` to reproduce
  pre-0.0.8 numbers. Not a constructor parameter; absent from `get_params`;
  does not survive `sklearn.base.clone`.

- **`interaction_gain_threshold` keeps its default of 4.0**, re-measured rather
  than assumed. On a five-feature design with a planted `x₀x₁` term over four
  seeds, the gain scale is preserved (planted pair 412 vs 398 nats at n=400;
  best null pair 0.35 vs 0.74 under a pure null), so the threshold carries
  over. Under a pure null the new penalty is marginally more conservative.

- **The block now has 22 columns rather than 25.** The penalty's affine null
  space — `a + bx_j + cx_k`, three directions — is dropped outright instead of
  being handed to `κ_jk`. Purification has already removed the affine part, so
  those directions carry the zero function, and dropping them is what makes the
  induced prior exactly `T_p Ω⁺ T_pᵀ/λ_jk`. Selecting instead on the design's
  column norms would not be invariant: two bases disagree about whether a
  direction is dropped or merely unpenalized, and the gain moves. Design rank
  is unchanged at 16; the six remaining unidentified directions contribute
  nothing to `edf`. The bilinear `x_j x_k` is *not* in the null space — it is
  the simplest genuine interaction, penalized through the mixed term.

### Added

- `TestTensorPenalty` replaces `TestTensorRankInvariance` in
  `tests/test_api.py`: checks the Kronecker form against an independent dense
  quadrature, the affine null space and the bilinear term's exclusion from it,
  the basis invariance itself, the ridge's *lack* of it (so the contrast
  cannot rot), the 22/16 column-and-rank structure, and `transform` round-trip.

### Known issues

- `benchmarks/RESULTS.md` is now three default changes stale (0.0.6 multiclass
  softmax, 0.0.7 spline roughness, 0.0.8 tensor roughness). Regenerate before
  quoting any of it.
- Conditional residual screening leaks: with a strong planted interaction
  present, the best *null* pair scores 6–7 nats, above the 4.0 threshold, under
  both the old and new penalties. The genuine pair ranks far above it, but this
  is a pre-existing property of scoring pairs against main-effect residuals,
  not something the penalty change introduces or fixes.

## [0.0.7] — 2026-07-31

### Changed

- **The spline roughness penalty is now the exact integrated squared second
  derivative**, `Ω_ab = ∫ B_a''(x) B_b''(x) dx`, replacing the plain
  coefficient-difference penalty `D₂ᵀD₂`. **Every fitted spline model changes**:
  `evidence_`, `edf_`, `lam_`, and — through them — interaction screening
  decisions and the pooled variants' edf budget.

  The difference penalty of Eilers and Marx is a roughness measure only for
  *equally spaced* knots, and SRAE places knots at empirical quantiles. Under
  non-uniform spacing it is implicitly weighted by knot density and
  under-penalizes curvature exactly where knots are dense. Holding basis, data,
  and engine fixed and varying only the penalty on one continuous feature:

  | x distribution | knot spacing ratio | Δ evidence | edf (plain → exact) |
  |---|---|---|---|
  | uniform | 2.0 | −2.2 | 10.1 → 9.5 |
  | normal | 9.6 | −2.4 | 9.3 → 9.0 |
  | gamma(2) | 24.8 | −25.3 | 12.6 → 8.9 |
  | lognormal | 129.8 | −21.9 | 12.7 → 8.2 |

  The exact penalty is better supported in every case and the gap widens with
  knot non-uniformity. Fitted curves move little (1–6% of the signal standard
  deviation); it is the reported quantities that shift.

  `Ω` is computed in closed form as `DᵀGD` — the derivative coefficient
  operator times the exact Gram matrix of the degree-(k−2) basis, integrated by
  Gauss–Legendre with `m+1` nodes per knot span, which is exact for degree-`m`
  products. This factorization also stays finite when two quantile knots nearly
  coincide, where a directly weighted difference penalty would blow up.

  Set the `SplineBlock.penalty` **class** attribute to `"difference"` to
  reproduce pre-0.0.7 numbers. As with `multiclass_link`, it is deliberately
  not a constructor parameter: absent from `get_params`, and it does not
  survive `sklearn.base.clone`.

  Tensor blocks are unaffected — they carry an isotropic ridge, not a
  difference penalty.

  Penalty eigenvalues are normalized to a maximum of 1. `Ω` carries units of
  `x⁻³`, so the same feature in millimetres rather than metres moves every
  eigenvalue by `1e9`; without normalizing, the EM update for `λ` — which
  starts at 1 — lands in a degenerate fixed point where the prior already
  dominates, the update returns `λ` unchanged, and the component collapses to
  a straight line regardless of the data. Only the overall scale is removed,
  so this reparametrizes `λ_j` without changing the prior, and it makes fits
  exactly invariant to the units of `x` (pinned by
  `test_fit_is_invariant_to_the_units_of_x` over 8 orders of magnitude). The
  consequence for readers of `summary()` is that `λ_j` is an inverse roughness
  variance up to a fixed per-feature constant, and is not comparable across
  features.

- **The penalty null space is now exactly the straight lines in raw `x`.**
  A consequence of the above, and a user-visible improvement: the direction
  governed by `kappa_j` is exactly linear on any knot spacing, verified to
  ~1e-12. Under the difference penalty on quantile knots it was only
  *trend-like*, which is what forced the hedged wording throughout the docs
  ("need not be exactly linear in raw x"). That hedging has been removed from
  the glossary, `inference.rst`, `interpretation.rst`, `model.rst`, and the
  `summary()` docstring, because it is no longer true.

### Fixed

- **Corrected the name "Demmler–Reinsch parametrization".** `SplineBlock`
  diagonalizes the penalty alone via `eigh`; canonical Demmler–Reinsch
  diagonalizes the penalty and the design inner product *jointly*, leaving
  `ZᵀZ` diagonal. Measured, the shipped `ZᵀZ` has off-diagonal entries the
  size of its diagonal, against 1.5e-6 for a true DR basis.

  This was a naming error only — no formula was wrong and no behaviour
  changed. The transformation was always written out correctly in
  `model.rst`, nothing relied on a DR-specific property (edf is the general
  `tr(I − AΣ)`, not `Σ 1/(1+λμ)`), and the shipped basis represents the
  standard P-spline prior family exactly, verified to 2e-4 relative. Renamed
  to "penalty eigenbasis" in `blocks.py`, `model.rst`, the glossary, and the
  README, with an explicit note on what the stronger property would have
  meant. The `interactions.rst` proposal of a Demmler–Reinsch tensor
  parametrization is unchanged — the term is used there in its correct joint
  sense, and the contrast is now stated.

### Added

- `TestRoughnessPenalty` in `tests/test_api.py`: verifies `βᵀΩβ` equals
  `∫(f'')²` against adaptive quadrature (agrees to ~1e-10 relative), that the
  null space is exactly the straight lines, that the `kappa_j` direction is
  linear in raw `x`, that quantile knots really are non-uniform enough for the
  distinction to matter, that the exact penalty beats differences on a skewed
  design, and that the `"difference"` escape hatch restores the old penalty.

### Known issues

- `benchmarks/RESULTS.md` is now two default changes stale (0.0.6 multiclass
  softmax, 0.0.7 roughness penalty). The penalty change affects regression
  figures too, not only classification. Regenerate before quoting any of it.

## [0.0.6] — 2026-07-27

### Changed

- **Dependency and interpreter floors raised to current releases.** numpy
  `>=2.5`, scipy `>=1.18`, pandas `>=3.0`, matplotlib `>=3.11`, scikit-learn
  `>=1.9`. numpy 2.5 and scipy 1.18 both require Python 3.12, so
  `requires-python` moves from `>=3.9` to `>=3.12` — below that the dependency
  set is unsatisfiable and an install fails at resolution rather than at import.
  The floors in the `sklearn`, `test`, and `benchmark` extras were raised to
  match; they previously named `scikit-learn>=1.2`, `pandas>=1.5`, and
  `matplotlib>=3.6`, which no longer described anything installable.

  This also retires a latent inconsistency: `tests/test_sklearn_compat.py`
  calls `check_estimator(..., on_fail=None)`, a parameter added in
  scikit-learn 1.6, while the declared floor was 1.2.

  The full suite (524 passed, 96 skipped), the warning-clean documentation
  build, and all 24 doctests were verified against these floors on both
  Python 3.12 and 3.13.

- **Default multiclass probability coupling is now a softmax over the stacked
  one-vs-rest log-odds** (`SRAEClassifier` and the pooled / scale-integrated
  classifiers). Reported multiclass probabilities differ from 0.0.5. The
  previous route moderated each head separately and divided by the row sum,
  which had two defects that only appear for `K > 2`: the row sum is not close
  to 1 (measured between roughly 0.39 and 3.56 on a 10-class problem), so
  normalising rescaled each row by a random quantity; and the per-head
  moderation shrinks toward 0.5, the neutral point for a binary question rather
  than the `1/K` of a `K`-class one, unevenly across heads. Renormalising
  cannot undo uneven shrinkage toward the wrong target.

  Binary classification and regression are unaffected — the binary path is not
  routed through the multiclass link, and both defects vanish at `K = 2`.

  Set the `multiclass_link` **class** attribute to `"normalized_ovr"` to
  reproduce pre-0.0.6 numbers. It is deliberately not a constructor parameter:
  it is absent from `get_params` and does not survive `sklearn.base.clone`, and
  it exists to reproduce previously published results, not to be tuned.

  The softmax route drops the per-head moderation rather than retargeting it,
  so it is a corrected *link*, not a corrected posterior. A joint multinomial
  Laplace approximation remains the principled fix.

### Fixed

- **`fit` no longer mutates the `feature_names` constructor parameter.** The
  resolved labels were written back onto the parameter, and because the write
  was guarded on `is None` it latched the *first* dataset: refitting on a
  differently-named frame produced a correctly fitted model that reported the
  previous dataset's labels. This also broke the scikit-learn estimator
  contract (`check_dont_overwrite_parameters`,
  `check_estimators_overwrite_params`), which meant `clone`, `Pipeline`, and
  `GridSearchCV` reuse could carry stale labels.
- **Multiclass fits now set `n_features_in_` and `feature_names_in_` on the
  pooled and scale-integrated classifiers.** The multiclass `fit` overrides in
  `pooled.py` and `scale_integration.py` never called
  `_set_sklearn_fit_attrs`, so multiclass `SRAEClassifierPooled`,
  `SRAEClassifierSI`, and `SRAEClassifierSIPooled` models exposed neither
  attribute at all.

Both fixes are pinned by regression tests in
`tests/test_sklearn_compat.py::TestFeatureNameHandling`, and the
parameter-handling subset of `sklearn.utils.estimator_checks.check_estimator`
is asserted to pass for all eight public estimators.

### Packaging

- Added `MANIFEST.in`. The sdist previously shipped `tests/test_*.py` without
  `tests/conftest.py` — setuptools' default file discovery does not pick up
  `conftest.py` — so the test suite in the source distribution collected
  nothing and downstream packagers could not verify a build.
- Added project URLs (homepage, documentation, source, changelog) to
  `pyproject.toml`; the PyPI page for 0.0.5 carried none.

### Continuous integration

- Added `.github/workflows/ci.yml`, running on every push and pull request.
  Previously the only workflow was `pypi.yml`, which runs on a published
  release, so nothing tested a change before it was merged. The workflow covers
  the full declared interpreter range (Python 3.12–3.13), builds the docs with
  `-W` to mirror the Read the Docs `fail_on_warning` setting, runs the
  documentation doctests, and unpacks the built sdist to confirm its test suite
  is runnable.
- `pypi.yml` now tests Python 3.12 and 3.13 — the ends of the range declared by
  `requires-python` — instead of 3.10 and 3.12, which left the floor unverified.

### Documentation

- **Documented the numerical status of the purified tensor basis.**
  `TensorBlock.fit` drops only constant columns, so its design is
  rank-deficient by construction (25 columns, rank 16 with quadratic spline
  marginals). The deficiency is structural — both marginals partition unity,
  so the 9-dimensional span that purification removes lies inside the tensor's
  own column space — and inert under the isotropic ridge: the free directions
  contribute zero edf and reduce the EM fixed point to the rank-reduced one,
  and the evidence depends on the design only through `Z Zᵀ/λ`. Redundant and
  prior-preserving rank-reduced representations agree to 2e-9 nats. No
  behaviour changed; the reasoning is now recorded in `srae/blocks.py` and
  `docs/user_guide/interactions.rst` rather than left implicit.
- **Documented that `interaction_gain_threshold` is defined against a basis
  convention.** A ridge on tensor B-spline coefficients is not invariant to a
  non-orthogonal reparameterization of the marginals; a moderate one moved the
  block-level gain by about 6 nats, more than the 4.0-nat default. The
  threshold is meaningful only relative to the shipped convention (quadratic
  B-spline marginals at `n_knots=2`, indicator marginals for nominal sides,
  purified then scaled to unit RMS, isotropic ridge). Recorded in the user
  guide, both `interaction_gain_threshold` docstrings, and the README's
  limitations. A basis-invariant construction is not attempted in 0.0.6.
- Corrected the reported doctest count (24, not 23) and the interpreter
  coverage claim in the README's verification status.
- Added this changelog.

### Added

- `TestTensorRankInvariance` in `tests/test_api.py` pins the invariances the
  tensor parametrization relies on — agreement under prior-preserving rank
  reduction and under orthogonal reparameterization — plus the rank-deficiency
  fact itself. These hold *because* the tensor prior is isotropic; introducing
  non-uniform `s_i` would break them, which is the point of asserting them.
  A fourth characterization test records the marginal-basis dependence above,
  so removing it becomes a deliberate change with a docs update.

### Known issues

- The bundled benchmark results in `benchmarks/RESULTS.md` were recorded with
  0.0.5 and have **not** been regenerated for 0.0.6. Because the default
  multiclass coupling changed, treat the classification figures as historical.
- `check_estimator` still fails the input-validation checks: sparse input,
  NaN/inf rejection, `NotFittedError` versus `RuntimeError`, and 1-D/2-D `y`
  handling. These are known gaps rather than silent misbehaviour.

## [0.0.5] — 2026-07-26

First release published to PyPI. Changes prior to 0.0.6 are not itemized here;
this changelog starts with 0.0.6.

[0.0.10]: https://github.com/kazilab/srae/releases/tag/v0.0.10
[0.0.9]: https://github.com/kazilab/srae/releases/tag/v0.0.9
[0.0.8]: https://github.com/kazilab/srae/releases/tag/v0.0.8
[0.0.7]: https://github.com/kazilab/srae/releases/tag/v0.0.7
[0.0.6]: https://github.com/kazilab/srae/releases/tag/v0.0.6
[0.0.5]: https://github.com/kazilab/srae/releases/tag/v0.0.5
