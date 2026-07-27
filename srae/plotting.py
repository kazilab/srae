"""Transparency plots: shape functions, interaction surfaces, complexity."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plot_shape_functions", "plot_interaction", "plot_evidence",
           "plot_importance"]


def plot_shape_functions(model, features=None, ncols=3, level=0.95,
                         figsize_per=(3.4, 2.6)):
    """Plot every main effect with a pointwise credible band.

    Parameters
    ----------
    model : fitted SRAE estimator
        A fitted regressor or binary classifier. Multiclass models do not
        expose a single set of shape functions; plot ``model.estimators_[k]``
        instead.

    features : list of str, default=None
        Component names to plot. When ``None``, every spline, linear and
        factor main effect is included, in fit order.

    ncols : int, default=3
        Number of subplot columns.

    level : float, default=0.95
        Nominal pointwise coverage of the shaded band.

    figsize_per : tuple of float, default=(3.4, 2.6)
        Size in inches of each subplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, with panel titles reporting each component's edf.

    See Also
    --------
    srae.SRAERegressor.shape_function : Underlying numeric values.

    Notes
    -----
    Bands are pointwise, not simultaneous, and condition on the estimated
    hyperparameters and the selected interaction set.
    """
    from scipy.stats import norm

    from .blocks import FactorBlock, LinearBlock, SplineBlock

    if features is None:
        features = [s.name for s, b in zip(model.specs_, model.blocks_)
                    if isinstance(b, (SplineBlock, LinearBlock, FactorBlock))]
    z = norm.ppf(0.5 + level / 2)
    n = len(features)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols,
                                      figsize_per[1] * nrows),
                             squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, name in zip(axes.ravel(), features):
        grid, mean, se = model.shape_function(name)
        lo, hi = mean - z * se, mean + z * se
        # Factor effects live on discrete levels; markers read better than a
        # dense fill_between band through meaningless intermediate codes.
        blk = next(b for s, b in zip(model.specs_, model.blocks_)
                   if s.name == name)
        if isinstance(blk, FactorBlock):
            ax.vlines(grid, lo, hi, color="tab:blue", lw=1.4, alpha=0.7)
            ax.plot(grid, mean, "o", color="tab:blue", ms=6)
        else:
            ax.fill_between(grid, lo, hi, alpha=0.25, lw=0, color="tab:blue")
            ax.plot(grid, mean, color="tab:blue", lw=1.6)
        ax.axhline(0, color="0.6", lw=0.7, ls=":")
        ax.set_title(f"f({name})   edf={model.edf_[name]:.1f}", fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("SRAE shape functions with credible bands", y=1.005)
    fig.tight_layout()
    return fig


def plot_interaction(model, pair_index=0, n_grid=60):
    """Heatmap of one purified pairwise interaction surface.

    Tensor blocks are residualized against their represented marginal and
    parent-main-effect spans on the training rows. The surface shown is the
    remaining pairwise contribution and is invariant to the values of the
    other features. Purification does not guarantee removal of main-effect
    structure outside those finite spans.

    Parameters
    ----------
    model : fitted SRAE estimator
        A fitted model with at least one entry in ``interactions_``.

    pair_index : int, default=0
        **Integer index into** ``model.interactions_``, not the dictionary
        stored there. Entries are ordered by screening gain.

    n_grid : int, default=60
        Resolution per axis of the evaluation grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, with axes labelled by the two feature names.

    Raises
    ------
    IndexError
        If ``pair_index`` is out of range, in particular when
        ``interactions_`` is empty.

    Examples
    --------
    >>> fig = plot_interaction(model, 0)             # doctest: +SKIP
    """
    from .blocks import TensorBlock

    info = model.interactions_[pair_index]
    j, k = info["pair"]
    tb = None
    spec = None
    for s, b in zip(model.specs_, model.blocks_):
        if isinstance(b, TensorBlock) and b.pair == (j, k):
            tb, spec = b, s
            break
    gx = np.linspace(model._xmin[j], model._xmax[j], n_grid)
    gy = np.linspace(model._xmin[k], model._xmax[k], n_grid)
    GX, GY = np.meshgrid(gx, gy)
    Xg = np.zeros((GX.size, len(model.feature_names_)))
    # midpoints for other features (purified surface is invariant to them)
    Xg[:] = (model._xmin + model._xmax) / 2
    Xg[:, j] = GX.ravel()
    Xg[:, k] = GY.ravel()
    Zg = model._design(Xg)
    surf = (Zg[:, spec.sl] @ model.beta_[spec.sl]).reshape(GX.shape)

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    m = ax.pcolormesh(GX, GY, surf, shading="auto", cmap="RdBu_r",
                      vmin=-np.abs(surf).max(), vmax=np.abs(surf).max())
    fig.colorbar(m, ax=ax, label=f"f({info['name']})")
    ax.set_xlabel(model.feature_names_[j])
    ax.set_ylabel(model.feature_names_[k])
    ax.set_title(f"Learned interaction {info['name']}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_evidence(model):
    """Trace of the log evidence across fitting iterations.

    Parameters
    ----------
    model : fitted SRAE estimator
        Any fitted model exposing ``evidence_history_``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The evidence trace.

    Notes
    -----
    A trace still climbing at the final iteration suggests raising
    ``max_iter``. Under the logistic Laplace approximation the trace need not
    be monotone, so small decreases are not by themselves a defect.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(model.evidence_history_, marker="o", ms=3)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log evidence")
    ax.set_title("Evidence maximization trace", fontsize=11)
    fig.tight_layout()
    return fig


def plot_importance(model, top=15):
    """Horizontal bar chart of component importances.

    Importance is the standard deviation of a component's contribution to the
    predictor on the training data.

    Parameters
    ----------
    model : fitted SRAE estimator
        Any fitted model whose ``summary()`` yields an ``importance`` column.

    top : int, default=15
        Number of highest-importance components to show.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The bar chart, ordered with the largest contribution at the top.

    Notes
    -----
    This is an in-sample dispersion, not a causal effect size and not a
    permutation importance; correlated features can trade off against one
    another.
    """
    df = model.summary().head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 0.34 * len(df) + 1.2))
    ax.barh(df["component"], df["importance"], color="tab:blue", alpha=0.85)
    ax.set_xlabel("importance  (sd of component contribution)")
    ax.set_title("SRAE component importance", fontsize=11)
    fig.tight_layout()
    return fig
