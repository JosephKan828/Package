# This program is private plot settings for matplotlib.
import numpy;
import matplotlib.pyplot as plt;

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


def get_masked_colormap(mask_range, cmap_name="RdBu_r"):
    """
    Returns a matplotlib colormap where values within mask_range are shown as white.

    Parameters:
    - mask_range : tuple (low, high), values within this range will be masked
    - cmap_name  : name of the base colormap to modify

    Returns:
    - cmap : a matplotlib colormap with .set_bad('white') for masked data
    - mask_func : a function that masks a 2D array based on the range
    """

    low, high = mask_range

    def mask_func(data):
        return np.ma.masked_inside(data, low, high)

    # Copy and modify the base colormap
    base_cmap = plt.get_cmap(cmap_name).copy()
    base_cmap.set_bad('white')

    return base_cmap, mask_func
