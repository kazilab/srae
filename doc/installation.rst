.. _installation:

============
Installation
============

Requirements
============

SRAE requires Python 3.12 or newer and the following runtime dependencies:

============== ==========  ====================================================
Package        Minimum     Used for
============== ==========  ====================================================
numpy          2.5         dense linear algebra, design matrices
scipy          1.18        B-spline bases, Cholesky factorization, eigen-solves
pandas         3.0         :meth:`~srae.SRAERegressor.summary` output
matplotlib     3.11        the ``plot_*`` helpers
scikit-learn   1.9         estimator protocol (``BaseEstimator``, mixins)
============== ==========  ====================================================

.. note::

   The Python floor follows from the dependency floors rather than from any
   language feature SRAE uses: numpy 2.5 and scipy 1.18 both require Python
   3.12, so an install on an earlier interpreter cannot resolve at all.

scikit-learn is a hard requirement: the estimators inherit from
:class:`sklearn.base.BaseEstimator` so that :func:`sklearn.base.clone`,
:class:`~sklearn.pipeline.Pipeline`, and
:class:`~sklearn.model_selection.GridSearchCV` work without adapters.

From source
===========

.. code-block:: bash

   python -m venv .venv
   . .venv/bin/activate
   pip install -e .

Optional extras
===============

.. code-block:: bash

   pip install -e ".[test]"        # pytest suite
   pip install -e ".[benchmark]"   # jupyter and comparison-model dependencies

Building the documentation
==========================

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html

Verifying the installation
==========================

.. code-block:: python

   >>> import numpy as np
   >>> from srae import SRAERegressor
   >>> rng = np.random.default_rng(0)
   >>> X = rng.normal(size=(200, 3))
   >>> y = np.sin(1.5 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.normal(size=200)
   >>> model = SRAERegressor(interactions=False).fit(X, y)
   >>> bool(model.score(X, y) > 0.8)
   True

The full test suite runs with:

.. code-block:: bash

   pytest
