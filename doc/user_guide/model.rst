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
training range.  Columns are mean-centered, and roughness is penalized by a
second-order difference penalty

.. math::

   \mathcal{P}(\boldsymbol{\beta})
   \;=\; \boldsymbol{\beta}^{\top} \mathbf{S} \boldsymbol{\beta},
   \qquad
   \mathbf{S} = \mathbf{D}_2^{\top} \mathbf{D}_2,

where :math:`\mathbf{D}_2` is the second-difference operator.  This is the
standard P-spline construction of Eilers and Marx [1]_.

.. _demmler_reinsch:

Demmler–Reinsch parametrization
-------------------------------

Rather than carrying a dense penalty matrix, SRAE rotates each block into the
basis that diagonalizes its penalty — the Demmler–Reinsch basis [2]_.  Writing
the eigendecomposition
:math:`\mathbf{S} = \mathbf{U}\operatorname{diag}(s)\mathbf{U}^{\top}`, the
block design becomes

.. math::

   \mathbf{Z}_j \;=\; (\mathbf{B} - \bar{\mathbf{B}})\,\mathbf{U},

after which directions that map to the (near-)zero function on the observed
data are dropped, and the retained columns are scaled to unit RMS.  A
second-order difference penalty has a two-dimensional zero-eigenvalue space.
Centering removes its constant *fitted contribution*, leaving a rank-one
trend-like contribution.  Because the eigendecomposition can return arbitrary
mixtures of the two zero-eigenvalue vectors before centering, the current
implementation can retain two collinear columns representing that one
contribution; this is redundant parametrization, not two distinct functions.
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
*separated*: :math:`\lambda_j` controls how wiggly :math:`f_j` may be, while
:math:`\kappa_j` is an ARD-style precision on the penalty null space — for a
second-order difference penalty, the centered trend-like contribution
described above.  With quantile-spaced knots this contribution is not
generally an exact straight line in raw :math:`x`.  A spline component can
therefore be driven to a low-dimensional, trend-like function (large
:math:`\lambda_j`, moderate :math:`\kappa_j`) or toward zero (both large), and
these outcomes are distinguishable in the reported hyperparameters.

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
freedom — see Wood [3]_.

.. rubric:: References

.. [1] P. H. C. Eilers and B. D. Marx, "Flexible smoothing with B-splines and
   penalties", *Statistical Science*, 11(2):89–121, 1996.

.. [2] A. Demmler and C. Reinsch, "Oscillation matrices with spline smoothing",
   *Numerische Mathematik*, 24:375–382, 1975.

.. [3] S. N. Wood, *Generalized Additive Models: An Introduction with R*,
   2nd ed., Chapman and Hall/CRC, 2017.
