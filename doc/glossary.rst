.. _glossary:

========
Glossary
========

.. glossary::
   :sorted:

   evidence
      The log marginal likelihood :math:`\log p(\mathbf{y} \mid
      \lambda, \kappa, \sigma^2)`, obtained by integrating the coefficients
      out of the joint density.  SRAE's estimation objective.  Exact for the
      Gaussian likelihood (:eq:`gaussian_evidence`), a Laplace approximation
      for the logistic one.  Reported as ``evidence_``.

   Occam factor
      The :math:`\tfrac12\sum_i \log a_i - \tfrac12\log\lvert\mathbf{H}\rvert`
      terms of the evidence.  They penalize models whose prior spreads mass
      over regions the posterior does not use, balancing fit against
      complexity without an explicit complexity grid.  Supported directions
      typically settle at a finite optimum; irrelevant ARD / null-space
      directions often optimize at a **boundary** (infinite or clipped
      precision), so a flat evidence trace need not imply fixed
      :math:`\lambda_j, \kappa_j`.

   effective degrees of freedom
   edf
      :math:`\operatorname{edf}_j = \sum_{i \in j}(1 - a_i \Sigma_{ii})`, the
      capacity a fitted model spends on component :math:`j`
      (:eq:`edf`).  Bounded by the block's column count.  ``edf`` near 1 means
      roughly one effective direction remains; what that direction represents
      depends on the block.  For a spline dominated by its penalty null space
      it is trend-like, but it need not be exactly linear in raw :math:`x`.
      ``edf`` near 0 indicates a component that is effectively switched off.

   Demmler–Reinsch parametrization
      Rotation of a penalized block into the eigenbasis of its penalty, making
      the prior precision matrix diagonal.  See :ref:`demmler_reinsch`.  This
      is what gives every hyperparameter a closed-form M-step.

   null-space direction
      A basis direction with penalty eigenvalue :math:`s_i = 0` — unpenalized
      by the roughness penalty and governed by :math:`\kappa_j` rather than
      :math:`\lambda_j`.  A second-order coefficient-difference penalty has a
      two-dimensional coefficient-space null space.  Centering removes the
      constant fitted contribution; the remaining fitted contribution is
      trend-like, but with quantile-spaced knots it is not generally exactly
      linear in raw :math:`x`.  In the current implementation an eigensolver
      can represent the centered null-space contribution with collinear
      retained columns, so interpret the fitted function and ``edf`` rather
      than the raw null-column count.

   penalized direction
      A basis direction with :math:`s_i > 0`, whose prior precision is
      :math:`\lambda_j s_i`.  Also called a *wiggly* direction.

   roughness precision
      :math:`\lambda_j`, the per-block precision multiplier on penalized
      directions.  Large values flatten curvature.  Reported in the ``lam``
      column of :meth:`~srae.SRAERegressor.summary`.

   null-space precision
      :math:`\kappa_j`, an ARD-style precision on the unpenalized directions of
      a block.  Large values remove the null-space contribution; the whole
      component is removed only when its penalized directions are also
      suppressed (or when it has no penalized directions).  Reported in the
      ``kappa`` column.

   purification
      Residualizing a tensor-product basis against an intercept and its two
      marginal bases and parent main-effect blocks (:eq:`purification`).
      Retained columns are orthogonal to those representable spans on the
      training data.  This makes the fitted decomposition identifiable within
      the chosen bases, but does not rule out leakage from unrepresented
      main-effect structure or finite-sample dependence.

   screening gain
      The evidence improvement of a purified tensor block fitted to the
      main-effects residuals, relative to a noise-only baseline
      (:eq:`screen_gain`).  Compared against
      ``interaction_gain_threshold``.  A conditional residual quantity, not an
      exact Bayes factor.

   working residuals
      The residual vector screening operates on: ordinary residuals for
      regression, gradient-space pseudo-residuals :math:`y_i - \mu_i` for
      classification.

   moderated probability
      A binary predicted probability that accounts for posterior uncertainty
      in the linear predictor via the probit approximation
      :math:`\operatorname{sigmoid}\big(\mu / \sqrt{1 + \pi\nu/8}\big)`
      (:eq:`moderated`), rather than plugging in the posterior mean.  The
      default multiclass link instead applies a softmax to one-vs-rest logits.

   Type-II maximum likelihood
   empirical Bayes
      Estimating hyperparameters by maximizing the evidence, then conditioning
      on that point estimate.  SRAE's default path.

   pooled stack
      The anti-overfitting layer of the ``*Pooled`` variants: conservative
      partial pooling, precision floors, soft pruning, a total-edf budget, and
      internal holdout *scale* calibration (not full held-out model
      validation).  See :ref:`axis_regularization`.

   total-edf budget
      A cap on :math:`\sum_j \operatorname{edf}_j`, enforced by bisection on a
      global precision multiplier.  Reported as ``edf_budget_`` against the
      realized ``total_edf_``.

   scale factor
      In the scale-integrated variants, the global multiplier :math:`f` applied to
      all block precisions: :math:`\lambda_j = f_\lambda \lambda_j^{*}`.  The
      quantity actually integrated over.  See :ref:`axis_hyperprior`.

   one-vs-rest
   OvR
      The multiclass construction: one independent binary SRAE per class.
      The default link couples the per-class one-vs-rest log-odds through a
      softmax.  The legacy ``normalized_ovr`` link moderates each binary head
      and then normalizes the rows.  Neither route is a jointly estimated
      multinomial model, so calibration should be checked.
