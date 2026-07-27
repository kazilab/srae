.. _interactions:

===========================
Interaction discovery
===========================

SRAE does not fit all :math:`\binom{p}{2}` pairwise surfaces.  It ranks
candidate pairs by an evidence gain computed on the residuals of the
main-effects fit and retains only those clearing a fixed threshold.

Purified tensor blocks
======================

A raw tensor-product basis overlaps the main effects: part of what it explains
is already attributable to :math:`f_j` and :math:`f_k`, which makes the
decomposition non-identifiable and the resulting surface impossible to read as
"the interaction" within the chosen basis representation.  SRAE therefore
*purifies* the tensor block.

Marginals :math:`\mathbf{B}_j` and :math:`\mathbf{B}_k` form the row-wise
Khatri–Rao product

.. math::

   \mathbf{T}_{i,\,(u,v)} \;=\; (\mathbf{B}_j)_{iu}\,(\mathbf{B}_k)_{iv},

which is then residualized against an intercept, the block's own marginal
bases, and the two parent main-effect blocks.  Writing
:math:`\mathbf{P} = [\,\mathbf{1} \;\; \mathbf{B}_j \;\; \mathbf{B}_k \;\;
\mathbf{Z}_j \;\; \mathbf{Z}_k\,]`,

.. math::
   :label: purification

   \mathbf{T}^{\text{pure}}
   \;=\; \mathbf{T} - \mathbf{P}\,\mathbf{P}^{+}\mathbf{T},

so every retained interaction column is orthogonal, on the training data, to
the span of its own main effects.  The projection coefficients
:math:`\mathbf{P}^{+}\mathbf{T}` are stored at fit time and reapplied at
predict time, so the same purification is used on new data.

**How each marginal is built depends on the parent block type.** Continuous
(or linear) features use low-resolution quadratic B-spline marginals.
Features fit with :class:`~srae.blocks.FactorBlock` use a *full* indicator
basis over the training levels (every level, not drop-one).  Indicators keep
the interaction surface invariant to how categories are numbered — a spline
over raw codes would not.  When both sides are factors the product is a
dummy–dummy tensor; factor × continuous mixes indicators with a spline
marginal.

Including :math:`\mathbf{B}_j` and :math:`\mathbf{B}_k` — not only the fitted
main-effect blocks :math:`\mathbf{Z}_j, \mathbf{Z}_k` — expands this
sample-level functional-ANOVA-style constraint to the tensor marginals.  The
two spans need not coincide: a
:class:`~srae.blocks.LinearBlock` contributes one column to
:math:`\mathbf{Z}_j` while a continuous tensor marginal is a spline basis,
and a factor main effect uses drop-one dummies while its tensor marginal keeps
all level indicators.  Any direction present in :math:`\mathbf{B}_j` but
absent from :math:`\mathbf{Z}_j` would otherwise let a pure main effect
survive purification and be screened as an interaction.
:math:`\mathbf{P}` is deliberately rank-deficient (spline marginals partition
unity; factor indicators plus an intercept are collinear), and the
pseudo-inverse handles this — the projection itself being unique.

The guarantee is limited to the spans in :math:`\mathbf{P}` and to
orthogonality on the training rows.  A main-effect shape outside the finite
marginal bases, finite-sample dependence, or approximation error in the
main-effects fit can still influence the residual screening score.  On new
data the stored projection is reapplied, but training-sample orthogonality is
not re-established.

Columns whose post-purification standard deviation is numerically zero are
dropped; the survivors are scaled to unit RMS and given a ridge penalty
(:math:`s_i = 1` for all directions), leaving one precision
:math:`\lambda_{jk}` per surface.

.. note::

   Purification makes :math:`f_{jk}` readable as the interaction remaining
   after projection against the represented marginal and parent-main-effect
   spans.  :func:`~srae.plot_interaction` shows that pairwise surface, which is
   invariant to the values of the other features, but it should not be read as
   proof that every possible main-effect contribution has been removed.  For
   nominal factors, coding-invariance of the interaction requires the
   indicator marginals above; ``feature_types='factor'`` alone used to fix
   only the main effect.

Working residuals
=================

Screening operates on residuals from the main-effects fit.  For regression
these are ordinary residuals, and for classification they are gradient-space
pseudo-residuals:

.. math::

   r_i^{\text{gauss}} = (y_i - \bar{y}) - \mathbf{z}_i^{\top}
                        \hat{\boldsymbol{\beta}},
   \qquad
   r_i^{\text{logit}} = y_i - \operatorname{sigmoid}\!\left(
                        \mathbf{z}_i^{\top}\hat{\boldsymbol{\beta}}\right).

The evidence gain
=================

For each candidate pair, a Gaussian empirical-Bayes model is fitted to the
residuals using only the purified tensor block, and its evidence is compared
against a noise-only baseline.  With :math:`\hat{v} = \operatorname{Var}(r)`,

.. math::
   :label: screen_gain

   \text{gain}_{jk}
   \;=\;
   \underbrace{\log p\!\left(\mathbf{r} \mid
        \mathbf{T}^{\text{pure}}_{jk}\right)}_{\text{fitted}}
   \;-\;
   \underbrace{\left(-\tfrac{n}{2}
        \left[\log(2\pi\hat{v}) + 1\right]\right)}_{\text{noise only}} .

Pairs are sorted by gain; those exceeding ``interaction_gain_threshold``
(default 4.0, on the log-evidence / nats scale) are kept, up to
``max_interactions`` (default 8).  The retained pairs and their gains are
exposed in ``interactions_``.  The model is then **refitted jointly** with main
effects and the selected tensor blocks together — the screening fit is used
only for ranking.

.. warning::

   :eq:`screen_gain` is a *conditional residual* marginal likelihood, not an
   exact Bayes factor for the full model.  It conditions on the main-effect fit
   and, under the logistic likelihood, additionally on a Gaussian
   approximation to gradient-space residuals.  The threshold is a structural
   setting, not a calibrated false-discovery control.

Candidate pre-ranking
=====================

When :math:`\binom{p}{2}` exceeds ``max_screen_pairs`` (default 40), fitting
every candidate is wasteful.  A cheap filter first ranks pairs by the absolute
correlation between the centered product :math:`\tilde{x}_j \tilde{x}_k` and
the residual vector,

.. math::

   \rho_{jk} \;=\;
   \frac{\left\lvert
      (\tilde{x}_j \odot \tilde{x}_k - \overline{\tilde{x}_j \odot \tilde{x}_k})
      ^{\top} \tilde{\mathbf{r}}
   \right\rvert}
   {\left\lVert \tilde{x}_j \odot \tilde{x}_k -
      \overline{\tilde{x}_j \odot \tilde{x}_k}\right\rVert \,
    \left\lVert \tilde{\mathbf{r}} \right\rVert},

and only the top ``max_screen_pairs`` proceed to the full evidence
computation.

.. admonition:: Known blind spot
   :class: warning

   This pre-filter is a *linear-in-the-product* screen.  Interactions that are
   symmetric about a feature's mean, masked by strong main effects, or
   otherwise poorly represented by a centered product can be discarded before
   they are ever scored.  Raise ``max_screen_pairs`` — up to
   :math:`\binom{p}{2}`, which disables the filter entirely — when a false
   negative is more costly than the extra computation.

Practical calibration
=====================

Screening power depends strongly on the likelihood.  Binary labels carry far
less information per observation than continuous responses, so the same
interaction needs appreciably more data to clear the same threshold.

On a smooth additive design with a planted :math:`x_0 x_1` term of comparable
strength, the pair is recovered at :math:`n = 80` under the Gaussian
likelihood, but is not selected at that size under the Bernoulli likelihood —
all variants return an empty ``interactions_`` until roughly :math:`n = 400`.

.. important::

   An empty ``interactions_`` on a small classification problem is evidence
   that screening lacked power, **not** evidence that no interaction exists.
   Before concluding absence, lower ``interaction_gain_threshold``, raise
   ``max_screen_pairs``, or check whether :math:`n` is simply too small.

Controlling the search
======================

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - Parameter
     - Default
     - Effect
   * - ``interactions``
     - ``"auto"``
     - ``False`` skips screening entirely and fits a purely additive model.
   * - ``interaction_gain_threshold``
     - ``4.0``
     - Minimum evidence gain in nats.  Lower admits more surfaces.
   * - ``max_interactions``
     - ``8``
     - Hard cap on retained surfaces, applied after thresholding.
   * - ``max_screen_pairs``
     - ``40``
     - Candidates scored by full evidence; above this the product-correlation
       pre-filter engages.
