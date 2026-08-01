.. _variants:

=========================
Estimator variants
=========================

The eight estimators are a :math:`2 \times 2` design over two independent axes,
crossed with the task.  They share the same model family, block constructors,
and screening algorithm and settings.  Their primary difference is how the
block hyperparameters :math:`(\lambda_j, \kappa_j, \sigma^2)` are treated.
The fitted result is not otherwise guaranteed to be identical: the
variant-specific main-effect fit determines the residuals passed to
interaction screening, so two variants can retain different interaction sets.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * -
     - **Point estimate**
     - **Integrated**
   * - **Type-II MLE**
     - :class:`~srae.SRAERegressor`,
       :class:`~srae.SRAEClassifier`
     - :class:`~srae.SRAERegressorSI`,
       :class:`~srae.SRAEClassifierSI`
   * - **Pooled stack**
     - :class:`~srae.SRAERegressorPooled`,
       :class:`~srae.SRAEClassifierPooled`
     - :class:`~srae.SRAERegressorSIPooled`,
       :class:`~srae.SRAEClassifierSIPooled`

``SRAE*SIPooled`` is exactly ``SRAE*SI(base="pooled")``; the class exists so
that :func:`sklearn.base.clone` and ``get_params`` see an explicit signature.

.. _axis_regularization:

Axis 1 — regularization
=======================

Type-II MLE (default)
---------------------

The maximum-evidence estimate of :ref:`em_updates`.  Nothing is added: each
supported block settles near a finite optimum of :eq:`gaussian_evidence`,
while irrelevant ARD directions may sit at the precision boundary (see
:ref:`gaussian_evidence`).  This is the reference behaviour and the one the
core method claims describe.

The pooled stack
----------------

:class:`~srae.SRAERegressorPooled` and :class:`~srae.SRAEClassifierPooled`
layer several anti-overfitting devices on top of that optimum, aimed at
small-:math:`n` / high-:math:`p` regimes where empirical Bayes is known to be
over-confident about its own hyperparameters.

**Conservative partial pooling.** Per-block precisions are shrunk toward a
common scale, but only ever *upward*:

.. math::

   \lambda_j^{\text{pool}}
   = \frac{r_j + 2\tau}{S_j + 2\tau/\bar{\lambda}},
   \qquad
   \lambda_j \leftarrow \max\!\left(\lambda_j^{\text{MLE}},\,
                                    \lambda_j^{\text{pool}}\right),

where :math:`\tau` is ``pool_strength``, :math:`S_j` the EM denominator, and
:math:`\bar{\lambda}` the 75th percentile of the per-block MLEs.  The
:math:`\max` makes the step one-sided: pooling can only *increase*
regularization, never relax it.  ``null_pool_strength`` controls the same
operation on the null-space precisions :math:`\kappa_j`.

**Precision floors.** Every precision is floored at
:math:`\texttt{floor\_scale}/n`, preventing any single block from becoming
arbitrarily flexible as :math:`n` shrinks.

**Soft pruning.** Blocks whose :math:`\operatorname{edf}` falls below
``prune_edf`` are pinned to a very large precision and the posterior is
recomputed, so a near-dead component cannot reabsorb capacity later.
``n_pruned_`` records how many were pinned.

**Total-edf budget.** A single global multiplier on all precisions is found by
bisection so that :math:`\sum_j \operatorname{edf}_j` respects a cap.  With
``max_total_edf="auto"`` that cap scales with sample size:

.. math::

   B_{\text{clf}} = \max\!\left(4,\;
        \min\!\left(0.50\sqrt{n},\; \frac{n}{3.5\log n}\right)\right),
   \qquad
   B_{\text{reg}} = \max\!\left(6,\;
        \min\!\left(0.85\sqrt{n},\; \frac{n}{2\log n}\right)\right).

The realized total and the cap are reported as ``total_edf_`` and
``edf_budget_``.

**Holdout scale calibration.** Finally a *single* global precision multiplier
is chosen by maximizing held-out log predictive density on an internal split
(``holdout_frac``, seeded by ``random_state``), reported as
``cal_scale_factor_``.  Classification uses a class-stratified split;
continuous regression targets trigger the routine's random-holdout fallback.
The Gaussian score uses :math:`z^\top\Sigma z + \sigma^2`; unlike the public
regression predictor, it does not add the small intercept-sampling term
:math:`\sigma^2/n`.  Logistic calibration scores moderated **binary**
probabilities.  In a multiclass fit this calibration happens separately
inside each one-vs-rest sub-model, whereas the parent estimator's default
prediction link couples its logits with a softmax.

.. warning::

   This is **not** held-out validation of the model.  Per-block scales,
   pruning, the edf budget, and residual variance are learned from *all*
   training rows; only the scalar multiplier is scored on the internal
   split.  Moreover, the multiplier from the main-effects fit changes the
   working residuals used for interaction screening, and calibration is run
   again if the selected interactions are jointly refitted.  The held-out
   rows have therefore influenced upstream fitted quantities, and the
   interaction set can be variant-dependent.  Treat ``cal_scale_factor_`` as
   a regularized point estimate, not an unbiased performance estimate.

   The split is drawn with ``random_state``.  An outer model-selection loop
   that reuses the same scheme will score the pooled variants on rows their
   calibration already saw.  Use a genuine outer split seeded independently.

.. _axis_hyperprior:

Axis 2 — hyperprior
===================

Point estimate (default)
------------------------

Predictions use the single maximizing :math:`\theta^{*} =
(\lambda^{*}, \kappa^{*}, \sigma^{2*})`.  Fast, and the usual empirical-Bayes
compromise.

Integrated (``SI``)
-------------------

Empirical Bayes is over-confident about :math:`\theta` itself at small
:math:`n`.  The ``SI`` variants keep the *relative* per-block scales fixed at
:math:`\theta^{*}` and put a posterior on **global multipliers**:

.. math::

   \lambda_j = f_\lambda \, \lambda_j^{*},
   \qquad
   \kappa_j  = f_\kappa  \, \kappa_j^{*},
   \qquad
   \sigma^2  = f_\sigma  \, \sigma^{2*},

with a weakly informative prior :math:`\log f \sim
\mathcal{N}(\texttt{log\_f\_prior\_mean}, \texttt{tau\_prior}^2)`.  A
random-walk Metropolis–Hastings sampler on :math:`\log f` targets
:math:`p(f \mid \mathbf{y}) \propto \exp\!\big(\text{evidence}(f)\big)\,
p(f)`, and the predictive distribution is the Monte Carlo average over draws.

What is integrated is the *capacity scale*, the direction that actually drives
overfitting.

.. important::

   This is **not full Bayes**, and the name is chosen to say so.  Fixing the
   relative per-block scales at :math:`\theta^{*}` makes the prior on
   :math:`(\lambda_j, \kappa_j)` itself a function of the data — the defining
   empirical-Bayes step.  Only :math:`f` carries a genuine prior and a genuine
   posterior.  The accurate description is *a scale mixture over global
   capacity multipliers, conditional on empirical-Bayes estimates of the
   relative per-block scales*.

   Sampling every block precision is not intractable: the log evidence is
   analytically differentiable in :math:`\log\lambda` and :math:`\log\kappa`,
   so a gradient-based sampler handles that dimension routinely.  The reasons
   for not doing it are that the random-walk sampler used here mixes poorly at
   that dimension, and — more substantively — that replacing
   :math:`\theta^{*}` with a genuine hyperprior reintroduces the prior
   sensitivity empirical Bayes exists to avoid.  At the sample sizes these
   variants target, the hyperprior scale moves results more than the sampler
   does.

   The two tasks are not equally addable.  Gaussian regression is conjugate
   and its evidence exact, so full Bayes over the hyperparameters is a
   feasible extension.  The logistic path already uses a Laplace
   approximation to the evidence, so full Bayes there would additionally
   require sampling :math:`\boldsymbol{\beta}` with no conjugacy.

.. note::

   The default ``min_f=1.0`` **truncates the prior** on :math:`\log f` to
   :math:`f \ge 1` — an informative prior encoding that at small :math:`n` the
   MAP is, if anything, under-regularized.  Truncation is a prior choice, not
   an approximation: the posterior is exact under the truncated prior.  Set
   ``min_f`` near 0 for an untruncated two-sided scale posterior.

Sampler adequacy
~~~~~~~~~~~~~~~~

The chain is short and the ``f_\lambda`` posterior is heavy-tailed, so the
sampler needs checking rather than trusting.  Each fit runs ``n_chains``
independent chains (default 4), adapts the proposal scale during burn-in
toward ~0.234 acceptance, and reports effective sample size and split-Rhat:

.. code-block:: python

   m = SRAERegressorSI().fit(X, y)
   m.min_ess_, m.max_rhat_          # worst case across f_lam, f_kap, f_sig

A ``RuntimeWarning`` is raised when ``ess_`` falls below 100 or ``rhat_``
exceeds 1.05.  The defaults (``n_samples=2000``, ``n_burn=1000``,
``n_chains=4``) aim to clear those bars on many smooth additive problems,
but that is not guaranteed — heavy tails in :math:`f_\lambda` and hard
designs can still warn. Always inspect ``min_ess_`` / ``max_rhat_``. At
:math:`n \approx 100` a default fit is on the order of
:math:`n_{\text{chains}}(n_{\text{burn}}+n_{\text{samples}}) \approx 12\,000`
posterior solves (often a few seconds, design-dependent).

.. warning::

   Settings far below the defaults will run without error and produce
   confident-looking intervals that are mostly sampler noise.  Monte Carlo
   error on a posterior mean scales as :math:`1/\sqrt{\text{ESS}}`, so an
   ESS of 4 puts it near half the posterior standard deviation.  Report
   ``min_ess_`` and ``max_rhat_`` alongside any uncertainty taken from these
   estimators.

For regression, :meth:`~srae.SRAERegressorSI.predict` averages both the mean
and the variance across draws, so the reported standard deviation obeys the law
of total variance,

.. math::

   \operatorname{Var}[y_\ast]
   = \mathbb{E}_{f}\!\left[\operatorname{Var}(y_\ast \mid f)\right]
   + \operatorname{Var}_{f}\!\left[\mathbb{E}(y_\ast \mid f)\right].

For binary classification, the moderated probabilities of :eq:`moderated` are
averaged across draws.  For multiclass classification with the default link,
one posterior logit draw from each one-vs-rest head is paired, each resulting
class vector is passed through a softmax, and those probability vectors are
averaged.  The legacy ``normalized_ovr`` link instead averages each head's
moderated binary probabilities before row normalization.

Diagnostics: ``accept_rate_`` (with ``adapt_step=True``, far from ~0.2–0.5
signals a difficult posterior), ``ess_`` / ``rhat_`` / ``min_ess_`` /
``max_rhat_``, ``mh_step_adapted_``, ``f_lam_mean_``, ``f_kap_mean_``,
``f_sig_mean_`` (regression), ``n_posterior_samples_``, ``n_chains_``, and
``map_evidence_`` for the stage-1 value. Multiclass SI reports worst-case
ESS / R-hat across one-vs-rest sub-models.

Every retained draw stores the scale factors used for ESS / R-hat. Full
coefficient arrays ``beta`` and ``Sigma`` are kept only for a thinned
subsample (at most 128 draws total) so memory stays bounded when the design
is large. The predictive Monte Carlo average uses that finite subsample: it
is still a valid estimator, but with larger Monte Carlo error than averaging
over every retained draw.

.. _variant_selection:

Choosing a variant
==================

.. admonition:: The evidence cannot rank these variants
   :class: caution

   ``evidence_`` is **not** comparable across the four.  On a fixed design,
   the pooled stack deliberately moves off the unpooled evidence optimum.
   Each fitted response in an ``SI`` variant reports the *mean log evidence*
   over a posterior restricted by default to :math:`f \ge 1`, rather than the
   value at :math:`f = 1`; a multiclass parent sums those one-vs-rest values.
   Screening can also produce different fitted designs across variants.  An
   evidence ranking therefore has no valid interpretation and will often
   favor the plain Type-II estimator, but no universal ordering is guaranteed.

   Compare variants by held-out predictive score only.

Measured behaviour
------------------

The only measurements this project stands behind are the ones a committed
script reproduces: ``benchmarks/RESULTS.md``, regenerated with
``python benchmarks/run_benchmarks.py``.  Four public datasets, five-fold
cross-validation, fixed seed.  Two patterns in it bear on the choice of
variant.

**The pooled stack costs accuracy on these datasets.**  It was behind the
plain Type-II estimator on every one — in :math:`R^2` for regression and in
held-out log-loss for classification.  Its capacity cap is deliberately
aggressive, so verify against :class:`~srae.SRAERegressor` /
:class:`~srae.SRAEClassifier` rather than assuming the pooled variant is
uniformly safer.  It was developed for small-sample settings that these four
datasets do not represent.

**Scale integration mostly buys calibration, not accuracy.**  ``SI`` tracked
the Type-II point estimate closely on regression :math:`R^2` while reporting
interval coverage marginally closer to nominal, and improved held-out log-loss
on both classification datasets — on the three-class problem substantially,
though with slightly lower accuracy.  That is the trade it is designed for:
integrating the hyperparameter scale widens intervals where the point estimate
was overconfident.

.. warning::

   Four datasets are not a benchmark suite.  They are public, low-dimensional
   and largely well-behaved — unlike correlated, high-dimensional real data
   such as spectra — and the baselines run at library defaults with no tuning,
   which flatters SRAE since it tunes its own shrinkage internally.  Treat this
   as orientation, not as a recommendation transferable to your problem, and
   validate on your own held-out data.

Practical guidance
------------------

- Start with :class:`~srae.SRAERegressor` / :class:`~srae.SRAEClassifier`.
- Consider ``SI`` when :math:`n` is small and calibrated uncertainty matters.
  Cost is roughly ``n_chains * (n_burn + n_samples)`` posterior solves per
  fit (about 12 000 with the shipped defaults), multiplied by the number of
  classes for multiclass. Always report ``min_ess_`` / ``max_rhat_``.
- Reach for the pooled variants only in genuinely small-:math:`n` /
  high-:math:`p` settings, and verify against the plain estimator — the
  capacity cap is aggressive.
- Because the axes are independent, ablate them independently rather than
  treating the four as an ordered ladder.
