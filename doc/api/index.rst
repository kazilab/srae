.. _api_reference:

=============
API reference
=============

This is the class and function reference for :mod:`srae`.  See the
:ref:`user_guide` for the definitions behind each quantity.

.. currentmodule:: srae

Regression estimators
=====================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SRAERegressor
   SRAERegressorPooled
   SRAERegressorSI
   SRAERegressorSIPooled

Classification estimators
=========================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SRAEClassifier
   SRAEClassifierPooled
   SRAEClassifierSI
   SRAEClassifierSIPooled

Plotting
========

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot_shape_functions
   plot_interaction
   plot_importance
   plot_evidence

Inference engines
=================

Low-level fitting routines used by the estimators.  Their signatures are
included for readers extending the package, but these functions are not the
stable estimator API.  Some have only signature-level reference material and
return implementation dictionaries; prefer the estimator classes for normal
use.

.. currentmodule:: srae.inference

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fit_gaussian_eb
   fit_logistic_eb

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BlockSpec

.. currentmodule:: srae.pooled

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fit_gaussian_eb_pooled
   fit_logistic_eb_pooled

.. currentmodule:: srae.scale_integration

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fit_gaussian_si
   fit_logistic_si

Basis blocks
============

.. currentmodule:: srae.blocks

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SplineBlock
   LinearBlock
   FactorBlock
   TensorBlock

.. autosummary::
   :toctree: generated/
   :template: function.rst

   make_block
   normalize_feature_type
