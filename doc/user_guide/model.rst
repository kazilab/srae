.. _model:

=========
The model
=========

Functional-ANOVA decomposition
==============================

SRAE fits an order-two functional-ANOVA model.  For the Gaussian case,

.. math::

   y_i \;=\; \beta_0 \;+\; \sum_{j=1}^{p} f_j(x_{ij})
   \;+\; \sum_{(j,k) \in \mathcal{S}} f_{jk}(x_{ij}, x_{ik})
   \;+\; \varepsilon_i,
   \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2),

and for binary classification the same additive predictor enters a
Bernoulli–logit likelihood,

.. math::

   \Pr(y_i = 1 \mid \mathbf{x}_i) \;=\;
   \operatorname{sigmoid}\!\big(\eta_i\big),
   \qquad
   \eta_i \;=\; \beta_0 + \sum_j f_j(x_{ij})
   + \sum_{(j,k)\in\mathcal{S}} f_{jk}(x_{ij}, x_{ik}).

Each :math:`f_j` and :math:`f_{jk}` is linear in basis coefficients, so the
whole predictor is :math:`\eta = \mathbf{Z}\boldsymbol{\beta}` for a stacked
design matrix :math:`\mathbf{Z}`.  Interpretability follows from this
structure: a fitted model *is* a list of one- and two-dimensional functions.

.. note::

   The regressor does not carry an explicit intercept column.  It centers the
   response, fits on :math:`y - \bar{y}`, and stores :math:`\bar{y}`.  Predictive
   standard deviations still include the sampling variance of that mean,
   :math:`\sigma^2 / n`, so the intercept is not treated as known.  The
   classifier does include an intercept block with a fixed, essentially flat
   prior that is excluded from hyperparameter updates.

Blocks
======

Every component contributes one *block* of columns to :math:`\mathbf{Z}`,
together with a vector of penalty eigenvalues :math:`s` that determines how
each direction is shrunk.

Spline blocks
-------------

For a continuous feature, :class:`~srae.blocks.SplineBlock` builds a cubic
B-spline basis :math:`\mathbf{B} \in \mathbb{R}^{n \times K}` on knots placed
at empirical quantiles, with clamped (constant) extrapolation outside the
training range.  Columns are mean-centered, and roughness is penalized by the
**exact integrated squared second derivative**

.. math::
   :label: roughness

   \mathcal{P}(\boldsymbol{\beta})
   \;=\; \boldsymbol{\beta}^{\top}\boldsymbol{\Omega}\,\boldsymbol{\beta}
   \;=\; \int \left(f''(x)\right)^2 \mathrm{d}x,
   \qquad
   \Omega_{ab} = \int B_a''(x)\,B_b''(x)\,\mathrm{d}x .

:math:`\boldsymbol{\Omega}` is computed in closed form as
:math:`\mathbf{D}^{\top}\mathbf{G}\mathbf{D}`, where :math:`\mathbf{D}` maps
coefficients to those of the second derivative in its degree-:math:`(k-2)`
B-spline basis and :math:`\mathbf{G}` is that basis's Gram matrix, integrated
exactly by Gauss–Legendre (products of degree-:math:`m` polynomials need only
:math:`m+1` nodes per knot span).

.. admonition:: Why not the plain difference penalty
   :class: note

   The classical P-spline construction of Eilers and Marx [1]_ replaces
   :eq:`roughness` with :math:`\mathbf{D}_2^{\top}\mathbf{D}_2` on the
   coefficients.  That is a roughness measure **only for equally spaced
   knots**, and SRAE places knots at empirical quantiles.  Under non-uniform
   spacing the plain penalty is implicitly weighted by knot density: it
   under-penalizes curvature exactly where knots are dense.

   Measured on one continuous feature with a planted smooth signal, holding
   basis, data, and engine fixed and varying only the penalty:

   .. list-table::
      :header-rows: 1
      :widths: 26 18 18 18 20

      * - :math:`x` distribution
        - knot spacing ratio
        - :math:`\Delta` evidence
        - edf (plain)
        - edf (exact)
      * - uniform
        - 2.0
        - −2.2
        - 10.1
        - 9.5
      * - normal
        - 9.6
        - −2.4
        - 9.3
        - 9.0
      * - gamma(2)
        - 24.8
        - −25.3
        - 12.6
        - 8.9
      * - lognormal
        - 129.8
        - −21.9
        - 12.7
        - 8.2

   The exact penalty is better supported in every case, and the gap widens
   with knot non-uniformity.  Fitted curves move very little (1–6% of the
   signal standard deviation) — it is the *reported* quantities, evidence and
   edf, that shift, which matters because edf is what the pooled variants
   budget and evidence is what interaction screening thresholds.

   Setting the ``SplineBlock.penalty`` class attribute to ``"difference"``
   restores the pre-0.0.7 behaviour for reproducing earlier results.  It is
   not a constructor parameter and does not survive
   :func:`sklearn.base.clone`.

.. _penalty_eigenbasis:

Penalty eigenbasis
------------------

Rather than carrying a dense penalty matrix, SRAE rotates each block into the
basis that diagonalizes its penalty.  Writing the eigendecomposition
:math:`\boldsymbol{\Omega} = \mathbf{U}\operatorname{diag}(s)\mathbf{U}^{\top}`,
the block design becomes

.. math::

   \mathbf{Z}_j \;=\; (\mathbf{B} - \bar{\mathbf{B}})\,\mathbf{U},

after which directions that map to the (near-)zero function on the observed
data are dropped, and the retained columns are scaled to unit RMS.

The eigenvalues are normalized to :math:`\max_i s_i = 1`.  :eq:`roughness`
carries units of :math:`x^{-3}`, so the same feature expressed in millimetres
rather than metres would move every eigenvalue by :math:`10^{9}`; normalizing
absorbs that into :math:`\lambda_j` and makes the fit exactly invariant to the
units of :math:`x`.  Only the overall scale is removed — the *relative*
eigenvalues carry the roughness weighting and are untouched — so this is a
reparametrization of :math:`\lambda_j`, not a change to the prior.  It does
mean :math:`\lambda_j` is an inverse roughness variance up to a fixed
per-feature constant, and so is not comparable across features.

.. note::

   This is *not* the Demmler–Reinsch parametrization [2]_, which diagonalizes
   the penalty and the design inner product **jointly** and therefore leaves
   :math:`\mathbf{Z}^{\top}\mathbf{Z}` diagonal as well.  Here only
   :math:`\boldsymbol{\Omega}` is diagonalized and
   :math:`\mathbf{Z}^{\top}\mathbf{Z}` is dense.  Nothing in SRAE relies on the
   stronger property: the evidence factorizes a full
   :math:`\mathbf{M}/\sigma^2 + \operatorname{diag}(a)` and :eq:`edf` is the
   general :math:`\operatorname{tr}(\mathbf{I} - \mathbf{A}\boldsymbol{\Sigma})`,
   not the Demmler–Reinsch form :math:`\sum_i (1 + \lambda \mu_i)^{-1}`.
   Documentation before 0.0.7 used the name loosely; the construction itself is
   unchanged.

The order-2 roughness penalty has a two-dimensional zero-eigenvalue space: the
straight lines in raw :math:`x`.  Centering removes the constant *fitted
contribution*, so exactly one function survives — a straight line in raw
:math:`x`.

The eigendecomposition returns an arbitrary mixture of the two zero-eigenvalue
vectors, and in practice one that leaves both columns non-constant, so a naive
norm filter would keep two perfectly collinear columns for that single
function.  Since 0.0.9 the null eigenspace is rotated first, separating the
identified direction from the zero function, and the block carries **one**
zero-penalty column.

.. note::

   The redundancy was inert, not wrong: two collinear coordinates sharing an
   isotropic :math:`\kappa_j` telescope to the same EM fixed point as one, so
   the evidence, edf, and fitted values were identical either way.  What it
   cost was reporting — ``n_coef`` was one too high, and the bound "edf cannot
   exceed the block's column count" was loose by one for no reason.
   Collapsing it rescales :math:`\kappa_j` (by exactly two when the two columns
   were equally scaled) without changing what :math:`\kappa_j` means.

Column scaling :math:`\mathbf{Z} \mapsto \mathbf{Z}/c` implies
:math:`\beta \mapsto \beta c`, so the eigenvalues transform as
:math:`s \mapsto s / c^{2}`.

The payoff is that the **prior precision matrix is diagonal**:

.. math::

   \beta_i \sim \mathcal{N}\!\left(0, \; a_i^{-1}\right),
   \qquad
   a_i =
   \begin{cases}
     \lambda_j \, s_i & s_i > 0 \quad \text{(penalized / \emph{wiggly})} \\[2pt]
     \kappa_j         & s_i = 0 \quad \text{(null space)}
   \end{cases}

Two consequences matter. First, the M-step for every hyperparameter has a
closed form (:ref:`em_updates`). Second, the two kinds of shrinkage are
*separated*: :math:`\lambda_j` controls how wiggly :math:`f_j` may be — and,
because :eq:`roughness` is a genuine roughness measure on any knot spacing,
it now shrinks curvature uniformly across the domain rather than in proportion
to knot density — while :math:`\kappa_j` is an ARD-style precision
on the penalty null space, which for the order-2 penalty is the centered linear
contribution described above.  Since 0.0.7 that contribution is *exactly* a
straight line in raw :math:`x`, on any knot spacing; under the pre-0.0.7
difference penalty it was only trend-like.  A spline component can therefore be
driven to a straight line (large :math:`\lambda_j`, moderate
:math:`\kappa_j`) or toward zero (both large), and these outcomes are
distinguishable in the reported hyperparameters.

Linear blocks
-------------

Under the default ``feature_types='auto'`` rule, features with at most
``max_linear_card`` (default 5) distinct values get a
:class:`~srae.blocks.LinearBlock`: a single standardized column with
:math:`s = 0`, i.e. one null-space direction governed by :math:`\kappa_j`.
This keeps binary and low-cardinality ordinal features from being handed a
spline basis they cannot support.

.. warning::

   A single standardized column is **linear in the numeric codes**. A
   low-cardinality feature that is nominal, or whose effect is non-monotone
   in those codes, is misspecified by this block and is fitted as flat ---
   the least-squares slope through a symmetric U-shape is approximately zero,
   and :math:`\kappa_j` then shrinks what remains. No warning is raised.

   Prefer ``feature_types='factor'`` (or a per-column dict entry) for those
   columns so they use a :class:`~srae.blocks.FactorBlock`. Alternatives:
   lower ``max_linear_card`` / set it to ``0`` so auto-dispatch hands them a
   spline, or one-hot encode externally. Cardinality alone cannot separate
   ordinal from nominal features, so no threshold is correct for every
   dataset.

   Tensor blocks purify against their own marginal bases (see
   :ref:`interactions`), so any omitted main-effect shape lying in those
   marginal spans is removed from the interaction basis on the training data.
   This substantially reduces leakage, but it is not a guarantee: structure
   outside those finite marginal spans, or finite-sample dependence, can still
   affect interaction screening.  Inspect the fit and validate selected
   interactions rather than assuming a misspecified main effect is harmless.

Factor blocks
-------------

A :class:`~srae.blocks.FactorBlock` models a nominal (or non-linear discrete)
feature with drop-one dummy coding. The last sorted training level is the
reference; the remaining :math:`K-1` columns are mean-centered, scaled to unit
RMS, and given a shared null-space ARD precision :math:`\kappa_j` (every
:math:`s_i = 0`). The block can represent arbitrary level means — including
U-shapes — without assuming order.

Declare factors explicitly via ``feature_types``::

   SRAERegressor(feature_types={"region": "factor", "dose": "spline"})
   SRAERegressor(feature_types=["factor", "auto", "linear"])

Aliases ``'categorical'``, ``'nominal'`` and ``'cat'`` resolve to
``'factor'``. After ``fit``, the requested types are on ``feature_types_``;
the realized blocks are on ``blocks_`` (``kind`` is also reported by
:meth:`~srae.SRAERegressor.summary`).

Shape functions for factors evaluate on the discrete training levels (not a
dense linspace), and :func:`~srae.plot_shape_functions` draws them as
point estimates with vertical error bars.

When a factor enters a retained pairwise interaction, the tensor side uses a
full indicator marginal over the same training levels (not a spline on the
numeric codes), so factor × continuous and factor × factor surfaces stay
coding-invariant.  See :ref:`interactions`.

Tensor blocks
-------------

Retained interactions use :class:`~srae.blocks.TensorBlock`, described in
:ref:`interactions`.  Continuous sides use low-resolution B-spline
marginals; factor sides use indicator marginals.  All retained directions are
penalized (:math:`s_i = 1`), so a tensor block is governed by a single ridge
precision :math:`\lambda_{jk}`.

The stacked design
==================

Concatenating blocks gives

.. math::

   \mathbf{Z} = [\,\mathbf{Z}_1 \;\; \mathbf{Z}_2 \;\; \cdots \;\;
                  \mathbf{Z}_{jk} \;\; \cdots\,],
   \qquad
   \mathbf{A} = \operatorname{diag}(a_1, \dots, a_q),

with the prior :math:`\boldsymbol{\beta} \sim
\mathcal{N}(\mathbf{0}, \mathbf{A}^{-1})`.  The mapping from a component name
to its column range is held in the block's ``BlockSpec``, which is what lets
:meth:`~srae.SRAERegressor.summary` and
:meth:`~srae.SRAERegressor.shape_function` isolate one component at a time.

Preprocessing expectations
==========================

SRAE does **not** impute, encode strings, or validate inputs beyond shape
checks. ``X`` must be a dense, fully numeric, finite 2-D array or DataFrame.
Missing values, string categoricals, and sparse matrices are the caller's
responsibility — put them in a :class:`~sklearn.pipeline.Pipeline` stage ahead
of the estimator. Numeric codes for nominal levels can stay in ``X`` and be
typed with ``feature_types='factor'``.

Feature scaling is *not* required: knots are quantile-based and every block is
internally standardized.

For the wider background on penalized-spline additive models — knot placement,
identifiability constraints, and the interpretation of effective degrees of
freedom — see Wood [3]_.  For the difference-penalty route to the same
non-uniform-knot problem, see Li and Cao [4]_.

.. rubric:: References

.. [1] P. H. C. Eilers and B. D. Marx, "Flexible smoothing with B-splines and
   penalties", *Statistical Science*, 11(2):89–121, 1996.

.. [2] A. Demmler and C. Reinsch, "Oscillation matrices with spline smoothing",
   *Numerische Mathematik*, 24:375–382, 1975.

.. [3] S. N. Wood, *Generalized Additive Models: An Introduction with R*,
   2nd ed., Chapman and Hall/CRC, 2017.

.. [4] Z. Li and J. Cao, "General P-splines for non-uniform B-splines",
   arXiv:2201.06808, 2022.  Derives knot-spacing-adjusted difference penalties
   for the non-uniform case; SRAE instead evaluates :eq:`roughness` exactly,
   which needs no difference approximation at all.
