.. _interpretation:

==============================
Interpreting a fitted model
==============================

A fitted SRAE object is a list of one- and two-dimensional functions plus a
capacity accounting.  This section maps each reporting method onto the
mathematical object it exposes.

Component report
================

:meth:`~srae.SRAERegressor.summary` returns one row per component, sorted by
importance:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Column
     - Definition
   * - ``component``
     - Component label, from ``feature_names_``.  Interactions appear as
       ``"a*b"``.
   * - ``kind``
     - ``"spline"``, ``"linear"``, ``"factor"``, or ``"tensor"``.
       Intercepts are not included as summary rows.
   * - ``n_coef``
     - Number of basis columns in the block.
   * - ``edf``
     - Effective degrees of freedom, :eq:`edf`.  Bounded by ``n_coef``.
   * - ``importance``
     - :math:`\operatorname{sd}\!\left(\mathbf{Z}_j
       \hat{\boldsymbol{\beta}}_j\right)` on the training data — the spread of
       that component's contribution to the predictor.
   * - ``lam``
     - Roughness precision :math:`\lambda_j`; ``NaN`` when the block has no
       penalized direction.
   * - ``kappa``
     - Null-space precision :math:`\kappa_j`; ``NaN`` when the block has no
       null direction.

Read ``edf`` together with the block ``kind``, ``lam``, and ``kappa``.  The
same edf does not imply the same shape for every block:

.. list-table::
   :header-rows: 1
   :widths: 31 18 51

   * - Pattern
     - edf
     - Reading
   * - spline: large :math:`\lambda`, moderate :math:`\kappa`
     - :math:`\approx 1`
     - Penalized curvature is suppressed; the retained contribution is
       low-dimensional and trend-like, but not necessarily exactly linear in
       raw :math:`x`.
   * - all applicable precisions large
     - :math:`\approx 0`
     - The component has been **switched off**.
   * - spline: moderate :math:`\lambda`
     - :math:`> 2`
     - Genuine **nonlinearity** is being supported by the data.
   * - factor or tensor
     - :math:`\approx 1`
     - Roughly one effective contrast or surface direction remains; this has
       no straight-line interpretation.

When edf is small the evidence optimum for that component is at
*infinite* precision: reported ``lam`` / ``kappa`` may still grow with
``max_iter`` after the evidence has flattened.  Prefer ``edf`` (stable) over
the raw precisions, and check ``at_boundary_`` for the list of such
components (edf below ``_BOUNDARY_EDF``, default 0.5 at the shipped
settings).  For a multiclass classifier these per-fit quantities live on each
object in ``estimators_`` rather than on the parent classifier.

.. note::

   ``importance`` is an in-sample dispersion, not a causal effect size and not
   a permutation importance.  Two components can trade off against each other
   when their features are correlated.

For multiclass models, ``summary()`` returns one section per class with an
added ``class`` column.

Shape functions
===============

:meth:`~srae.SRAERegressor.shape_function` evaluates a single main effect on a
grid and returns its posterior mean and pointwise standard error:

.. math::

   \hat{f}_j(g) = \mathbf{G}\hat{\boldsymbol{\beta}}_j,
   \qquad
   \operatorname{se}(g) = \sqrt{
     \operatorname{diag}\!\left(\mathbf{G}\,
       \boldsymbol{\Sigma}_{jj}\,\mathbf{G}^{\top}\right)},

where :math:`\mathbf{G}` is the block's basis evaluated on the grid.  A
pointwise band is :math:`\hat{f}_j \pm z\,\operatorname{se}`.

.. code-block:: python

   >>> grid, mean, se = model.shape_function("dose")
   >>> lower, upper = mean - 2 * se, mean + 2 * se

By default the grid spans the observed training range of the feature.  Outside
that range the spline basis uses **clamped (constant) extrapolation**, so
predictions flatten rather than diverge — safe, but not evidence about the
unobserved region.

.. important::

   These are *pointwise* conditional bands, not simultaneous bands, and they
   condition on the estimated hyperparameters and the selected interaction set.
   Do not read them as covering the entire curve simultaneously at the nominal
   level.

For multiclass models pass the class label:

.. code-block:: python

   >>> grid, mean, se = mc.shape_function("dose", cls=mc.classes_[0])

Interactions
============

``interactions_`` is a list of dictionaries, each holding ``pair`` (the feature
index tuple), ``name``, and ``screen_gain`` — the value of
:eq:`screen_gain` for that pair.  Because tensor blocks are purified
(:eq:`purification`), the plotted surface is the pairwise contribution left
after projection against the represented marginal and parent-main-effect spans
on the training data.  It is invariant to the other features' values, but
purification does not prove that main-effect structure outside those finite
spans could not influence screening.

.. code-block:: python

   >>> [(i["name"], round(i["screen_gain"], 1)) for i in model.interactions_]
   [('x0*x1', ...)]
   >>> fig = plot_interaction(model, 0)   # integer index into interactions_

Evidence trace
==============

For a regressor or binary classifier, ``evidence_`` is the final log marginal
likelihood and ``evidence_history_`` is the per-iteration trace, plotted by
:func:`~srae.plot_evidence`.  It is a useful convergence diagnostic — a trace
still climbing at the last iteration suggests raising ``max_iter``.  A *flat*
trace is weaker evidence of stability: irrelevant ARD directions can still
crawl toward the precision bound after the evidence has levelled (see
:ref:`gaussian_evidence`).

A multiclass parent exposes ``evidence_`` as the sum across one-vs-rest
sub-models, but does not expose a combined ``evidence_history_``.  Plot or
inspect the trace of an individual object in ``estimators_`` instead.

.. warning::

   ``evidence_`` is comparable across models **only** when they share the same
   estimation path, likelihood, and data.  It is not comparable across the four
   variants (see :ref:`variant_selection`), and for classification it is a
   Laplace approximation that need not increase monotonically.

Plotting helpers
================

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Function
     - Shows
   * - :func:`~srae.plot_shape_functions`
     - Grid of all main effects with credible bands.
   * - :func:`~srae.plot_interaction`
     - Heatmap of one purified interaction surface.  Takes an **integer
       index** into ``interactions_``.
   * - :func:`~srae.plot_importance`
     - Components ranked by ``importance``.
   * - :func:`~srae.plot_evidence`
     - Evidence trace across iterations for a regressor, binary classifier, or
       one selected multiclass sub-model.

Feature labels
==============

Component labels are resolved at fit time into ``feature_names_``, with
precedence: explicit ``feature_names`` argument, then DataFrame columns, then
``x0 … xp``.

The ``feature_names`` *constructor parameter* is never written to by ``fit``
(required by the scikit-learn estimator contract), so refitting on a
differently-named frame relabels correctly:

.. code-block:: python

   >>> m = SRAERegressor().fit(df_a, y)        # doctest: +SKIP
   >>> m.feature_names_                        # doctest: +SKIP
   ['alpha', 'beta']
   >>> m = m.fit(df_b, y)                      # doctest: +SKIP
   >>> m.feature_names_                        # doctest: +SKIP
   ['red', 'green']

``feature_names_in_`` follows the scikit-learn convention: it is set only when
``X`` carries column names, and is removed when a later fit does not.

Variant-specific diagnostics
============================

A single-response pooled fit additionally exposes ``total_edf_``,
``edf_budget_``, ``n_pruned_``, ``cal_scale_factor_``,
``edf_scale_factor_``, and ``a_floor_``.  A multiclass pooled parent aggregates
``total_edf_``, ``cal_scale_factor_``, and ``loo_scale_factor_``; inspect each
object in ``estimators_`` for the remaining per-class pooled diagnostics.

A single-response scale-integrated fit exposes ``samples_``, ``accept_rate_``,
``f_lam_mean_``, ``f_kap_mean_``, ``f_sig_mean_`` for regression,
``map_evidence_``, and ``n_posterior_samples_``, plus ``ess_``, ``rhat_``,
``min_ess_``, ``max_rhat_``, ``n_chains_``, and ``mh_step_adapted_``.  A
multiclass SI parent aggregates the acceptance and scale means, sample count,
MAP evidence, and worst-case ESS/R-hat diagnostics, but coefficient
``samples_`` and other per-class details stay on its ``estimators_``.
See :ref:`variants`.

.. warning::

   Any uncertainty taken from a scale-integrated variant — a predictive
   interval, a scale-factor mean, ``evidence_`` — is only as good as the
   sampler that produced it.  **Check ``min_ess_`` and ``max_rhat_`` before
   quoting any of it**, and report both alongside the number::

      m = SRAERegressorSI().fit(X, y)
      m.min_ess_, m.max_rhat_        # e.g. (112.4, 1.029)

   Monte Carlo error on a posterior mean scales as
   :math:`1/\sqrt{\text{ESS}}`, so at an ESS of 4 — which the pre-0.0.5
   defaults produced — the sampler noise is roughly half the posterior
   standard deviation, and the reported interval mostly reflects the chain
   rather than the posterior.  A ``RuntimeWarning`` is raised at fit time
   when ``ess_`` drops below 100 or ``rhat_`` exceeds 1.05; do not silence it
   for anything you intend to interpret.
