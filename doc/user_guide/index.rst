.. _user_guide:

==========
User guide
==========

This guide states the model, the estimation objective, and the screening
procedure precisely, then maps each mathematical object onto the attribute that
reports it.

.. toctree::
   :numbered:
   :maxdepth: 2

   model
   inference
   interactions
   variants
   interpretation

Notation
========

Throughout, :math:`n` is the number of samples and :math:`p` the number of
input features.  The stacked design matrix is :math:`\mathbf{Z} \in
\mathbb{R}^{n \times q}`, formed by concatenating per-component blocks, and
:math:`\boldsymbol{\beta} \in \mathbb{R}^{q}` collects all basis coefficients.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Symbol
     - Meaning
   * - :math:`f_j`
     - main effect of feature :math:`j`, a smooth univariate function
   * - :math:`f_{jk}`
     - purified pairwise interaction surface for features :math:`(j,k)`
   * - :math:`\mathcal{S}`
     - set of interaction pairs retained by screening
   * - :math:`\mathbf{Z}_j`
     - design block for component :math:`j` (columns of :math:`\mathbf{Z}`)
   * - :math:`s_i`
     - penalty eigenvalue of basis direction :math:`i`; :math:`s_i = 0`
       marks a null-space direction
   * - :math:`\lambda_j`
     - roughness precision of block :math:`j` (penalized directions)
   * - :math:`\kappa_j`
     - null-space (ARD) precision of block :math:`j`
   * - :math:`a_i`
     - prior precision of coefficient :math:`i`; :math:`\mathbf{A} =
       \operatorname{diag}(a)`
   * - :math:`\sigma^2`
     - Gaussian residual variance
   * - :math:`\boldsymbol{\Sigma}`
     - posterior covariance of :math:`\boldsymbol{\beta}`
   * - :math:`\operatorname{edf}_j`
     - effective degrees of freedom of component :math:`j`
