.. _inference:

=========
Inference
=========

SRAE treats the shrinkage parameters as latent quantities estimated from the
data by maximizing the marginal likelihood — the *evidence* — rather than by
cross-validated grid search.  This section states that objective, its
stationary conditions, and the quantities derived from it.

.. _gaussian_evidence:

Gaussian evidence
=================

With :math:`\mathbf{y} \sim \mathcal{N}(\mathbf{Z}\boldsymbol{\beta},
\sigma^2 \mathbf{I})` and :math:`\boldsymbol{\beta} \sim
\mathcal{N}(\mathbf{0}, \mathbf{A}^{-1})`, the coefficients integrate out in
closed form.  Define the posterior precision and moments

.. math::

   \mathbf{H} \;=\; \frac{\mathbf{Z}^{\top}\mathbf{Z}}{\sigma^{2}}
                    + \mathbf{A},
   \qquad
   \hat{\boldsymbol{\beta}} \;=\;
       \mathbf{H}^{-1} \frac{\mathbf{Z}^{\top}\mathbf{y}}{\sigma^{2}},
   \qquad
   \boldsymbol{\Sigma} \;=\; \mathbf{H}^{-1}.

The log marginal likelihood is then

.. math::
   :label: gaussian_evidence

   \log p(\mathbf{y} \mid \lambda, \kappa, \sigma^2)
   \;=\;
   -\frac{n}{2}\log(2\pi\sigma^{2})
   \;-\; \frac{\lVert \mathbf{y} - \mathbf{Z}\hat{\boldsymbol{\beta}} \rVert^{2}}
              {2\sigma^{2}}
   \;-\; \tfrac{1}{2}\hat{\boldsymbol{\beta}}^{\top}\mathbf{A}
                     \hat{\boldsymbol{\beta}}
   \;+\; \tfrac{1}{2}\sum_i \log a_i
   \;-\; \tfrac{1}{2}\log\lvert \mathbf{H} \rvert .

The last two terms are the Occam factor of the evidence framework [5]_.
Increasing a precision :math:`a_i`
raises :math:`\tfrac12\log a_i` but also inflates the misfit terms, so
supported directions typically settle at a **finite** optimum of
:eq:`gaussian_evidence` — the balance between fit and complexity is a
property of the objective, not of a tuning grid.

That statement does **not** mean every precision has an interior maximum.
Irrelevant ARD / null-space directions commonly drive
:math:`\kappa_j` (or a weakly supported :math:`\lambda_j`) toward the
upper precision clip while the evidence approaches a **boundary** limit.
A flat ``evidence_history_`` for an individual fit therefore does not
guarantee that the reported precisions have stopped moving; inspect ``lam`` /
``kappa`` in :meth:`~srae.SRAERegressor.summary` (and ``n_iter_``) when a
component looks switched off.  After ``fit``, components with edf below
``_BOUNDARY_EDF`` (default 0.5) are listed on ``at_boundary_``: treat their
reported precisions as iteration cut-offs, not as data-determined values.  For
a multiclass classifier, these trace, iteration, edf, and boundary attributes
live on the one-vs-rest objects in ``estimators_`` rather than on the parent.

All factorizations use a Cholesky decomposition of :math:`\mathbf{H}`, so
:math:`\log\lvert\mathbf{H}\rvert` is obtained as
:math:`2\sum_i \log L_{ii}` without forming an explicit inverse.

.. _em_updates:

EM updates
==========

Treating :math:`\boldsymbol{\beta}` as the latent variable, each M-step is
available in closed form.  For block :math:`j` with :math:`r_j` penalized and
:math:`m_j` null-space directions,

.. math::

   \lambda_j \;\leftarrow\;
     \frac{r_j}{\displaystyle\sum_{i:\,s_i>0}
                s_i\left(\hat{\beta}_i^{2} + \Sigma_{ii}\right)},
   \qquad
   \kappa_j \;\leftarrow\;
     \frac{m_j}{\displaystyle\sum_{i:\,s_i=0}
                \left(\hat{\beta}_i^{2} + \Sigma_{ii}\right)},

and, for the Gaussian model only,

.. math::

   \sigma^{2} \;\leftarrow\;
     \frac{\lVert \mathbf{y} - \mathbf{Z}\hat{\boldsymbol{\beta}} \rVert^{2}
           + \operatorname{tr}\!\left(\mathbf{Z}\boldsymbol{\Sigma}
                                      \mathbf{Z}^{\top}\right)}{n}.

The :math:`\Sigma_{ii}` terms are what distinguish these from naive
plug-in updates: they carry the posterior uncertainty of each coefficient, so a
poorly determined direction is not allowed to look small merely because its
point estimate is.

Iteration alternates posterior computation and these updates until the relative
change in evidence falls below ``tol`` or ``max_iter`` is reached.  The
evidence trace for each fitted response is kept in ``evidence_history_`` and
``n_iter_`` records its iteration count.  Because null ARD directions may
still crawl toward the precision bound after the evidence has levelled off,
``tol`` alone is not a certificate that every hyperparameter is stable.

.. note::

   After the loop terminates, the posterior is recomputed once at the final
   hyperparameters.  Without that refresh, ``beta_``, ``Sigma_``,
   ``evidence_``, ``edf_``, and regression-only ``sigma2_`` would correspond
   to *different* hyperparameter values — the returned quantities for each
   fitted response are synchronized by construction.

.. _laplace:

Logistic case: Laplace approximation
====================================

Under a Bernoulli–logit likelihood the coefficient integral is not analytic.
SRAE uses a Laplace approximation [6]_ about the penalized MAP estimate, found by
damped Newton / IRLS iterations with a backtracking line search on

.. math::

   \ell(\boldsymbol{\beta})
   \;=\; \sum_i \Big[ y_i \eta_i - \log\!\left(1 + e^{\eta_i}\right) \Big]
         \;-\; \tfrac{1}{2}\boldsymbol{\beta}^{\top}\mathbf{A}
                            \boldsymbol{\beta}.

With :math:`\mu_i = \operatorname{sigmoid}(\eta_i)` and
:math:`\mathbf{W} = \operatorname{diag}\!\big(\mu_i(1-\mu_i)\big)`, the
curvature at the mode is

.. math::

   \mathbf{H} \;=\; \mathbf{Z}^{\top}\mathbf{W}\mathbf{Z} + \mathbf{A},
   \qquad
   \boldsymbol{\Sigma} \;=\; \mathbf{H}^{-1},

giving the approximate log evidence

.. math::

   \log p(\mathbf{y} \mid \lambda, \kappa)
   \;\approx\;
   \ell(\hat{\boldsymbol{\beta}})
   + \tfrac{1}{2}\sum_i \log a_i
   - \tfrac{1}{2}\log\lvert\mathbf{H}\rvert ,

which is then maximized by the same posterior-moment fixed-point updates.

The quadratic prior term appears **once**, inside
:math:`\ell(\hat{\boldsymbol{\beta}})` as defined above; the
:math:`\tfrac{p}{2}\log 2\pi` of the Gaussian prior cancels against the same
factor in the Laplace volume.  In :func:`~srae.inference.fit_logistic_eb` the
helper ``_map_logistic`` returns the *unpenalized* log-likelihood, so the
implementation spells the same quantity as
``ll - 0.5 * beta @ (a * beta) + 0.5 * sum(log a) - 0.5 * logdetH``.

.. warning::

   This path is approximate.  The monotone-ascent guarantee of the Gaussian EM
   argument does **not** carry over, and an individual fit's evidence trace in
   ``evidence_history_`` may be non-monotone.

.. _edf:

Effective degrees of freedom
============================

The capacity a fitted model spends on component :math:`j` is reported as

.. math::
   :label: edf

   \operatorname{edf}_j \;=\; \sum_{i \in \text{block } j}
       \left(1 - a_i \Sigma_{ii}\right),

the block-restricted trace of :math:`\mathbf{I} - \mathbf{A}\boldsymbol{\Sigma}`.
It is bounded by the number of columns in the block and decreases toward zero
as shrinkage tightens.  An :math:`\operatorname{edf}` near 1 means roughly one
effective direction remains, not universally that the component is a straight
line.  For a spline dominated by its penalty null space, that direction *is* a
straight line in raw :math:`x` — exactly so since 0.0.7, on any knot spacing;
for factor and tensor blocks it has a different interpretation.  An :math:`\operatorname{edf}` near 0 means the component has
effectively been removed.

Total capacity :math:`\sum_j \operatorname{edf}_j` is what the pooled variants
constrain through an explicit budget (see :ref:`variants`).

.. _moderated_probabilities:

Moderated probabilities for binary classification
==================================================

For binary classification, plugging the posterior mean into the link would
ignore coefficient uncertainty and produce over-confident probabilities.
SRAE instead integrates the link against the Gaussian posterior of the linear
predictor.  With :math:`\mu =
\mathbf{z}^{\top}\hat{\boldsymbol{\beta}}` and :math:`\nu =
\mathbf{z}^{\top}\boldsymbol{\Sigma}\mathbf{z}`, the probit approximation to
that integral gives

.. math::
   :label: moderated

   \Pr(y = 1 \mid \mathbf{z})
   \;\approx\;
   \operatorname{sigmoid}\!\left(
      \frac{\mu}{\sqrt{1 + \pi\nu/8}}
   \right),

which is what :meth:`~srae.SRAEClassifier.predict_proba` returns for a binary
fit.  The denominator shrinks predictions toward :math:`0.5` exactly where the
model is least certain.  This is MacKay's moderated output [4]_.

.. _multiclass_link:

Multiclass probabilities
------------------------

Since 0.0.10 the default ``multiclass_link="joint"`` refits the structure
discovered one-vs-rest as a **joint multinomial model** and predicts from its
Laplace posterior, moderating toward the :math:`K`-class neutral point:

.. math::
   :label: moderated_mc

   \mathbf{p} \;=\; \operatorname{softmax}\!\left(
      \frac{\boldsymbol{\eta}}{\sqrt{1 + \pi\bar{\nu}/8}}
   \right),
   \qquad
   \bar{\nu} \;=\; \operatorname*{mean}_{k<l}
      \operatorname{Var}\!\left(\eta_k - \eta_l\right),

with the class logits carried in a sum-to-zero contrast basis.  A *common*
factor per row, rather than one per class, is what keeps the link independent
of class labelling: adding a constant to every logit leaves a softmax
unchanged.  At :math:`K = 2` both :eq:`moderated_mc` and the joint engine
reduce exactly to their binary counterparts.

Two legacy routes remain, for reproducing published results.
``multiclass_link="softmax"`` (the 0.0.6-0.0.9 default) softmaxes the
one-vs-rest log-odds with no moderation; ``"normalized_ovr"`` (to 0.0.5)
moderates each head and divides by the row sum.

.. warning::

   Coherent row sums are not calibrated probabilities.  Measured held-out on
   synthetic multiclass data over five seeds, the ``softmax`` route is the
   *worst* of the three — 2-4 times the expected calibration error of
   ``normalized_ovr``, and higher log-loss, at every :math:`K` and :math:`n`
   tried.  Neither one-vs-rest route yields a joint posterior: independent
   binary fits leave the cross-class Hessian blocks at zero, so no coherent
   covariance between class surfaces exists to moderate with.

The pooled and scale-integrated variants override the multiclass fit with
machinery that has no joint analogue yet, so they fall back to
``normalized_ovr``.  The scale-integrated classifier instead preserves its
scale uncertainty by softmaxing paired posterior logit draws and averaging the
resulting probability vectors.

Predictive distribution for regression
======================================

:meth:`~srae.SRAERegressor.predict` with ``return_std=True`` reports

.. math::

   \operatorname{Var}\!\left[y_\ast \mid \mathbf{z}_\ast\right]
   \;=\;
   \underbrace{\mathbf{z}_\ast^{\top}\boldsymbol{\Sigma}\mathbf{z}_\ast}
             _{\text{parameter uncertainty}}
   \;+\;
   \underbrace{\sigma^{2}}_{\text{observation noise}}
   \;+\;
   \underbrace{\sigma^{2}/n}_{\text{intercept sampling variance}},

where the last term comes from estimating the response mean
:math:`\bar y` (the regressor fits on :math:`y - \bar y` and adds
:math:`\bar y` back at predict time).  It is small —
:math:`1/n` of the noise variance — but treating
:math:`\bar y` as known omitted the only fitted quantity that had no
posterior variance.  :class:`~srae.SRAERegressorPooled` and
:class:`~srae.SRAERegressorSI` use the same intercept term; SI further
averages over draws of the global scale factors via the law of total
variance.

:meth:`~srae.SRAERegressor.predict_interval` turns this into an
equal-tailed Gaussian interval at the requested level.

.. admonition:: What these intervals do *not* cover
   :class: warning

   All uncertainty summaries still condition on the estimated
   hyperparameters (relative per-block scales), the selected interaction
   set :math:`\mathcal{S}`, and the fixed basis.  The intercept term only
   accounts for uncertainty in :math:`\bar y`.  The ``SI`` variants
   integrate global capacity scales, but they do not integrate every
   :math:`\lambda_j, \kappa_j` and they never propagate *selection*
   uncertainty in :math:`\mathcal{S}`.  Empirical-Bayes GAM intervals are
   therefore mildly optimistic at small :math:`n`; see
   :meth:`~srae.SRAERegressor.predict_interval` for measured coverage.

Numerical safeguards
====================

All precisions are clipped to :math:`[10^{-10}, 10^{12}]`, the Bernoulli
log-likelihood uses the stable :func:`~numpy.logaddexp` form, the sigmoid is
evaluated branch-wise by sign, and IRLS weights are floored at
:math:`10^{-10}` to keep :math:`\mathbf{H}` positive definite under separation.

.. rubric:: References

.. [4] D. J. C. MacKay, "The evidence framework applied to classification
   networks", *Neural Computation*, 4(5):720–736, 1992.

.. [5] D. J. C. MacKay, "Bayesian interpolation", *Neural Computation*,
   4(3):415–447, 1992.

.. [6] C. M. Bishop, *Pattern Recognition and Machine Learning*, chapters 3–4,
   Springer, 2006.
