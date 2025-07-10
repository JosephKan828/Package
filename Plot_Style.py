# This program is private plot settings for matplotlib.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def apply_custom_plot_style(fontsize=24, use_latex=False):
    """
    Applies a custom set of plot settings to matplotlib.

    Parameters
    ----------
    fontsize : int, optional
        The font size to use for all text in the plot. Defaults to 12.
    linewidth : int, optional
        The line width to use for all lines in the plot. Defaults to 2.
    use_latex : bool, optional
        Whether to use LaTeX to render text in the plot. Defaults to False.

    Returns
    -------
    None
    """
    style_dict = {
        "font.size": fontsize,
        "font.family": "serif",
        "axes.titlesize": fontsize + 8,
        "axes.labelsize": fontsize + 4,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "text.usetex": use_latex,
        "axes.grid": True,
        "grid.alpha": 0.5,
    }
    plt.rcParams.update(style_dict)


def colorlist():
    return [
        "#000000",  # black
        "#E69F00",  # orange (colorblind-friendly)
        "#56B4E9",  # sky blue (colorblind-friendly)
        "#009E73",  # bluish green (colorblind-friendly)
        "#F0E442",  # yellow (works well on white)
        "#0072B2",  # blue (colorblind-friendly)
        "#D55E00",  # vermilion (colorblind-friendly red-orange)
        "#CC79A7"   # reddish purple
    ]


def insert_white_into_colormap(cmap_name='RdBu_r', white_range=(-0.1, 0.1), N=256):
    """
    Create a new ListedColormap based on an existing colormap, but with a specified
    value range replaced by white.

    Parameters:
    -----------
    cmap_name : str
        Name of the base matplotlib colormap (e.g., 'RdBu_r', 'viridis', etc.)
    white_range : tuple of float
        Value range to be shown as white (e.g., (-0.1, 0.1)). Must be within [-1, 1].
    N : int
        Number of color levels in the final colormap.

    Returns:
    --------
    new_cmap : ListedColormap
        The modified colormap with white inserted.
    """

    base = plt.get_cmap(cmap_name, N)
    colors = base(np.linspace(0, 1, N))

    # Normalize white_range to [0, 1] assuming colormap is symmetric around 0
    x = np.linspace(-1, 1, N)
    mask = (x >= white_range[0]) & (x <= white_range[1])

    colors[mask] = [1.0, 1.0, 1.0, 1.0]  # RGBA for white

    new_cmap = ListedColormap(colors)
    return new_cmap
