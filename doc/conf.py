"""Sphinx configuration for the SRAE documentation.

Follows the scikit-learn documentation conventions: numpydoc-formatted
docstrings, an autosummary-driven API reference, and a narrative user guide
carrying the mathematical definitions.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from srae import _metadata  # noqa: E402

# -- Project information ----------------------------------------------------

project = _metadata.PROJECT_NAME
author = _metadata.AUTHOR
copyright = _metadata.COPYRIGHT
version = _metadata.VERSION
release = _metadata.VERSION

# -- General configuration --------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
root_doc = "index"

# -- numpydoc / autodoc -----------------------------------------------------

# Let autosummary render the member tables; numpydoc's own class-member
# listing would duplicate them (this is what scikit-learn does).
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = "all"

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "none"
add_function_parentheses = False

# -- Cross-project references -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- HTML output ------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = f"{_metadata.FULL_NAME} {release}"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "titles_only": False,
}

# Shared LaTeX macros for the model equations.
mathjax3_config = {
    "tex": {
        "macros": {
            "bbeta": r"\boldsymbol{\beta}",
            "bSigma": r"\boldsymbol{\Sigma}",
            "bZ": r"\mathbf{Z}",
            "by": r"\mathbf{y}",
            "bA": r"\mathbf{A}",
            "bH": r"\mathbf{H}",
            "edf": r"\operatorname{edf}",
            "tr": r"\operatorname{tr}",
        }
    }
}

nitpicky = False
