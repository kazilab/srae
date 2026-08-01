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

The surface is penalized by its integrated squared second derivatives, and the
block is rotated into the eigenbasis of that penalty, leaving one precision
:math:`\lambda_{jk}` per surface.

The roughness penalty
---------------------

.. math::
   :label: tensor_roughness

   \mathcal{P}(\boldsymbol{\beta})
   \;=\; \iint \left[
      \left(\tfrac{\partial^2 f}{\partial x_j^2}\right)^2
      + 2\left(\tfrac{\partial^2 f}{\partial x_j \partial x_k}\right)^2
      + \left(\tfrac{\partial^2 f}{\partial x_k^2}\right)^2
   \right] \mathrm{d}x_j\,\mathrm{d}x_k ,

the thin-plate-style combination of pure and mixed second derivatives.  In
Kronecker form it is
:math:`\boldsymbol{\Omega}_j \otimes \mathbf{G}_k
+ 2\,\boldsymbol{\Omega}^{(1)}_j \otimes \boldsymbol{\Omega}^{(1)}_k
+ \mathbf{G}_j \otimes \boldsymbol{\Omega}_k`,
built from the one-dimensional derivative and Gram matrices of each margin, so
it costs three small products.  Each margin's knots are rescaled to
:math:`[0,1]` first: the three terms carry different powers of the domain
length, so on raw knots their relative weighting would drift as a feature was
re-expressed in different units — the standard non-invariance of an isotropic
thin-plate penalty under differential scaling of covariates.

A single :math:`\lambda_{jk}` is kept rather than one per margin.  The
anisotropic form would double the hyperparameters of every candidate pair
during screening, for a surface that is discarded unless it clears the gain
threshold.

.. _tensor_invariance:

Why the gain no longer depends on the basis
-------------------------------------------

:eq:`tensor_roughness` is a functional of the fitted surface, not of its
coordinates.  Under a change of marginal basis
:math:`\mathbf{B}_j \mapsto \mathbf{B}_j\mathbf{R}`,
:math:`\mathbf{B}_k \mapsto \mathbf{B}_k\mathbf{S}` — spanning exactly the same
surfaces — the purified design and the penalty transform together,

.. math::

   \mathbf{T}^{\text{pure}} \mapsto \mathbf{T}^{\text{pure}}\mathbf{W},
   \qquad
   \boldsymbol{\Omega} \mapsto \mathbf{W}^{\top}\boldsymbol{\Omega}\mathbf{W},
   \qquad
   \mathbf{W} = \mathbf{R} \otimes \mathbf{S},

leaving the induced prior
:math:`\mathbf{T}^{\text{pure}}\boldsymbol{\Omega}^{+}\mathbf{T}^{\text{pure}\top}`
unchanged.  The isotropic ridge used before 0.0.8 had no such transformation
law: :math:`\mathbf{I}` is the identity in whatever coordinates it is written,
so reparametrizing the marginals moved the prior and with it the gain.

Measured on a continuous × continuous pair, holding data and engine fixed:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - penalty
     - basis dependence of the gain
     - edf
   * - isotropic ridge (to 0.0.7)
     - 1.7 nats
     - 11.6
   * - roughness (0.0.8)
     - :math:`8 \times 10^{-11}` nats
     - 8.3

The roughness penalty is also better supported on that pair (about 12 nats)
while spending less capacity, which is what the lower edf records.

Rank deficiency of the purified block
-------------------------------------

Purification removes
:math:`\operatorname{span}[\mathbf{1}, \mathbf{B}_j, \mathbf{B}_k]`, which —
because both spline marginals partition unity — lies *inside* the tensor's own
column space with dimension :math:`5 + 5 - 1 = 9`.  The raw 25 columns
therefore carry design rank 16.

Three of those nine are the penalty's null space, the affine surfaces
:math:`a + b x_j + c x_k`.  Note the bilinear :math:`x_j x_k` is **not** among
them: it is the simplest genuine interaction, and the mixed-derivative term is
what penalizes it.  The affine directions are dropped outright rather than
handed to :math:`\kappa_{jk}` — purification has already removed the affine
part, so they carry the zero function, and dropping them is what makes the
induced prior exactly
:math:`\mathbf{T}^{\text{pure}}\boldsymbol{\Omega}^{+}\mathbf{T}^{\text{pure}\top}/\lambda_{jk}`.

.. important::

   Selecting on the design's *column norms* instead — which coordinates happen
   to look null — would break the invariance above.  Two bases would disagree
   about whether a direction was dropped or merely unpenalized, and the gain
   would move.  This is why the shipped block selects on penalty eigenvalues.

The block therefore has 22 columns of design rank 16.  The remaining six
directions are penalized but unidentified; they contribute nothing to
:math:`\operatorname{edf}`, which stays bounded by the design rank because
those directions lie in the null space of the design's Gram, and the proper
prior keeps the posterior computable.

These properties — exactness of :eq:`tensor_roughness` against an independent
quadrature, the affine null space, the invariance itself, and the ridge's lack
of it — are pinned by ``TestTensorPenalty`` in ``tests/test_api.py``.

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

What the threshold is and is not calibrated against
---------------------------------------------------

Since 0.0.8 the gain no longer depends on how the marginals are parametrized
(:ref:`tensor_invariance`), so ``interaction_gain_threshold`` is no longer tied
to a coefficient convention.  It does remain tied to the *structural* settings:
the marginal resolution (``n_knots=2`` quadratic splines, indicator marginals
for nominal sides), the purification in :eq:`purification`, and the isotropic
combination in :eq:`tensor_roughness`.

The default of 4.0 was re-measured across the change rather than assumed.  On a
five-feature design with one planted :math:`x_0 x_1` term, averaging four
seeds, comparing the retired ridge against the roughness penalty:

.. list-table::
   :header-rows: 1
   :widths: 14 12 12 20 20

   * - penalty
     - :math:`n`
     - signal
     - gain, planted pair
     - gain, best null pair
   * - ridge
     - 200
     - 0.0
     - −0.43
     - 0.74
   * - roughness
     - 200
     - 0.0
     - −0.56
     - 0.35
   * - ridge
     - 400
     - 2.0
     - 397.6
     - 6.13
   * - roughness
     - 400
     - 2.0
     - 412.4
     - 7.14

The gain scale is preserved, so 4.0 carries over unchanged; under a pure null
the roughness penalty is if anything marginally more conservative.

.. warning::

   The "best null pair" column exceeds 4.0 once a strong real interaction is
   present, under both penalties.  Residual leakage from a large planted term
   inflates the scores of pairs sharing a feature with it.  This is a
   pre-existing property of conditional residual screening, not a consequence
   of the penalty change, and it is why ``max_interactions`` matters: the
   planted pair ranks far above the leakage, but the leakage can still clear a
   fixed threshold.

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
