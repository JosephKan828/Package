# This program is private plot settings for matplotlib.
import matplotlib.pyplot as plt

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