"""SRAE: Self-Regularizing Additive Estimator.

An interpretable empirical-Bayes additive model for tabular data, with spline
main effects and evidence-screened pairwise interactions. Continuous shrinkage
parameters are estimated internally; structural basis and screening settings
remain explicit choices.
"""

from ._metadata import AUTHOR as __author__
from ._metadata import EMAIL as __email__
from ._metadata import VERSION as __version__
from .model import SRAEClassifier, SRAERegressor
from .pooled import SRAEClassifierPooled, SRAERegressorPooled
from .scale_integration import (
    SRAEClassifierSI,
    SRAEClassifierSIPooled,
    SRAERegressorSI,
    SRAERegressorSIPooled,
)
from .plotting import (plot_evidence, plot_importance, plot_interaction,
                       plot_shape_functions)

__all__ = [
    "SRAERegressor",
    "SRAEClassifier",
    "SRAERegressorPooled",
    "SRAEClassifierPooled",
    "SRAERegressorSI",
    "SRAEClassifierSI",
    "SRAERegressorSIPooled",
    "SRAEClassifierSIPooled",
    "plot_shape_functions",
    "plot_interaction",
    "plot_evidence",
    "plot_importance",
]
