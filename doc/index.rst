.. _srae_docs:

==========================================
SRAE: Self-Regularizing Additive Estimator
==========================================

**SRAE** is an evidence-screened empirical-Bayes additive model.  It fits an
interpretable order-two functional-ANOVA model

.. math::

   f(\mathbf{x}) \;=\; \beta_0
   \;+\; \sum_{j} f_j(x_j)
   \;+\; \sum_{(j,k)\,\in\,\mathcal{S}} f_{jk}(x_j, x_k)

with penalized-spline main effects and a small set of automatically screened
pairwise tensor interactions.  The roughness precisions
:math:`\lambda_j`, the null-space precisions :math:`\kappa_j`, and (for Gaussian
regression) the residual variance :math:`\sigma^2` are estimated from a
marginal-likelihood objective rather than by cross-validated grid search.

Every fitted main effect is plottable with a pointwise credible band; selected
interactions are plottable as two-dimensional mean surfaces.  Model capacity is
reported as effective degrees of freedom per component.

.. note::

   Basis resolution, candidate-pair caps, screening thresholds, and maximum
   interaction counts remain **structural settings** chosen by the user.  Only
   the continuous shrinkage parameters are estimated internally.  See
   :ref:`limitations`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quick_start

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   glossary

Estimator overview
==================

Eight estimators span a :math:`2 \times 2` design over two independent axes,
crossed with the regression / classification task.

.. list-table::
   :header-rows: 1
   :widths: 22 22 28 28

   * - Regularization
     - Hyperprior
     - Regression
     - Classification
   * - Type-II MLE
     - point estimate
     - :class:`~srae.SRAERegressor`
     - :class:`~srae.SRAEClassifier`
   * - pooled stack
     - point estimate
     - :class:`~srae.SRAERegressorPooled`
     - :class:`~srae.SRAEClassifierPooled`
   * - Type-II MLE
     - integrated
     - :class:`~srae.SRAERegressorSI`
     - :class:`~srae.SRAEClassifierSI`
   * - pooled stack
     - integrated
     - :class:`~srae.SRAERegressorSIPooled`
     - :class:`~srae.SRAEClassifierSIPooled`

The **regularization** axis selects how the block hyperparameters
:math:`(\lambda_j, \kappa_j)` are estimated; the **hyperprior** axis selects
whether those hyperparameters are used as a point estimate or integrated over.
See :ref:`variants` for when each is appropriate.

.. _limitations:

Scope and limitations
=====================

- The model is restricted to smooth main effects and *selected pairwise*
  interactions; no three-way or higher terms.
- Interaction discovery is greedy and conditional on the main-effect fit.  The
  screening score is a conditional residual marginal likelihood, **not** an
  exact Bayes factor for the full model — especially under the logistic
  likelihood.
- The product-correlation pre-ranking used when the candidate set exceeds
  ``max_screen_pairs`` can miss interactions that are symmetric, masked, or
  poorly represented by a centered product.
- Uncertainty summaries condition on the estimated hyperparameters, the
  selected interaction set, and the fixed basis.  They do **not** propagate
  interaction-selection uncertainty.
- Multiclass classification is one-vs-rest, not a jointly estimated
  multinomial model.  By default, independently fitted one-vs-rest log-odds
  are coupled through a softmax; the legacy row-normalized one-vs-rest link is
  available only when selected explicitly.  In either case, calibration
  should be checked.
- The logistic path is approximate (Laplace); evidence monotonicity is not
  guaranteed by the Gaussian EM argument.
- The implementation uses dense linear algebra and does not accept sparse
  input, missing values, or non-numeric columns.

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
