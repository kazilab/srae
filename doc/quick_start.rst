.. _quick_start:

===========
Quick start
===========

Regression
==========

.. code-block:: python

   >>> import numpy as np
   >>> from srae import SRAERegressor
   >>> rng = np.random.default_rng(0)
   >>> X = rng.normal(size=(400, 4))
   >>> y = (np.sin(1.5 * X[:, 0]) + X[:, 1] ** 2
   ...      + 1.5 * X[:, 0] * X[:, 1] + 0.3 * rng.normal(size=400))
   >>> model = SRAERegressor(n_knots=10, interactions="auto").fit(X, y)
   >>> pred = model.predict(X)
   >>> lo, hi = model.predict_interval(X, level=0.90)

The fitted object reports its own structure.  ``edf`` is the effective degrees
of freedom spent on each component and ``importance`` is the standard deviation
of that component's contribution on the training data:

.. code-block:: python

   >>> model.summary()             # doctest: +SKIP
     component    kind  n_coef   edf  importance      lam  kappa
   0     x0*x1  tensor       9  4.31        1.42     0.09    NaN
   1        x1  spline      10  3.02        1.05    12.44   0.31
   2        x0  spline      10  2.87        0.98    18.10   0.44

Which pairwise interactions survived screening:

.. code-block:: python

   >>> [(i["name"], round(i["screen_gain"], 1)) for i in model.interactions_]
   [('x0*x1', ...)]

Classification
==============

:class:`~srae.SRAEClassifier` covers binary and, through a one-vs-rest
ensemble, multiclass problems.  For a **binary** fit, ``predict_proba`` returns
*moderated* probabilities — the Laplace posterior variance is folded into the
link (see :ref:`moderated_probabilities`):

.. code-block:: python

   >>> from srae import SRAEClassifier
   >>> y_bin = (y > np.median(y)).astype(int)
   >>> clf = SRAEClassifier(n_knots=8, interactions="auto").fit(X, y_bin)
   >>> proba = clf.predict_proba(X)
   >>> proba.shape
   (400, 2)

For multiclass models each class keeps its own fully transparent additive
model, exposed through ``estimators_``.  ``shape_function`` takes a ``cls``
argument, while ``summary`` returns rows for all classes.  The default
prediction link applies a softmax to the stacked
one-vs-rest log-odds; it does not apply the binary moderation formula to each
head.  The scale-integrated classifier instead softmaxes paired posterior
logit draws and averages the resulting probability vectors:

.. code-block:: python

   >>> y_mc = np.digitize(y, np.quantile(y, [1/3, 2/3]))
   >>> mc = SRAEClassifier(interactions=False).fit(X, y_mc)
   >>> len(mc.estimators_) == len(mc.classes_)
   True
   >>> grid, mean, se = mc.shape_function(0, cls=mc.classes_[0])

Inspecting components
=====================

Every main effect is a one-dimensional function with a pointwise standard
error, evaluated on a grid:

.. code-block:: python

   >>> grid, mean, se = model.shape_function(0)
   >>> band = (mean - 2 * se, mean + 2 * se)

Convenience plotting helpers wrap this:

.. code-block:: python

   >>> from srae import (plot_shape_functions, plot_interaction,
   ...                   plot_importance, plot_evidence)
   >>> fig = plot_shape_functions(model)          # doctest: +SKIP
   >>> fig = plot_interaction(model, 0)           # index into interactions_
   >>> fig = plot_importance(model)               # doctest: +SKIP
   >>> fig = plot_evidence(model)                 # doctest: +SKIP

.. warning::

   :func:`~srae.plot_interaction` takes an **integer index** into
   ``interactions_``, not the dictionary stored there.

Using pandas
============

When ``X`` is a :class:`~pandas.DataFrame`, its column names become the
component labels used by ``summary()``, ``shape_function()``, and the plotting
helpers:

.. code-block:: python

   >>> import pandas as pd
   >>> Xdf = pd.DataFrame(X, columns=["age", "dose", "ph", "temp"])
   >>> m = SRAERegressor(interactions=False).fit(Xdf, y)
   >>> m.feature_names_
   ['age', 'dose', 'ph', 'temp']
   >>> grid, mean, se = m.shape_function("dose")

scikit-learn integration
========================

All eight estimators implement the standard estimator protocol:

.. code-block:: python

   >>> from sklearn.model_selection import cross_val_score, GridSearchCV
   >>> from sklearn.pipeline import Pipeline
   >>> from sklearn.preprocessing import StandardScaler
   >>> scores = cross_val_score(
   ...     SRAERegressor(interactions=False), X, y, cv=5)
   >>> pipe = Pipeline([("scale", StandardScaler()),
   ...                  ("srae", SRAERegressor(interactions=False))]).fit(X, y)
   >>> search = GridSearchCV(
   ...     SRAERegressor(interactions=False),
   ...     {"n_knots": [6, 10]}, cv=3).fit(X, y)

Because SRAE estimates its shrinkage parameters internally, a grid search is
only needed over *structural* settings such as ``n_knots`` or
``interaction_gain_threshold`` — not over penalty strengths.

Choosing a variant
==================

Start with :class:`~srae.SRAERegressor` / :class:`~srae.SRAEClassifier`.  The
pooled and scale-integrated variants target small-sample optimism and are described
in :ref:`variants`, which also reports measured trade-offs.
