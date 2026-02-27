import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def add_text_with_contrast(ax, matrix, cmap_name, fmt="{:.2e}"):
    """
    Annotates each cell in a matrix heatmap with its value, choosing a text 
    color (black or white) for maximum contrast with the cell background.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes object containing the plot to annotate.
    matrix : np.ndarray
        2D array of values to annotate.
    cmap_name : str
        Name of the matplotlib colormap used in imshow.
    fmt : str, optional
        Format string for displaying the values inside cells.
    """
    norm = mpl.colors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
    cmap = plt.get_cmap(cmap_name)
    for (i, j), val in np.ndenumerate(matrix):
        r, g, b, _ = cmap(norm(val))
        brightness = 0.299*r + 0.587*g + 0.114*b
        text_color = "black" if brightness > 0.6 else "white"
        ax.text(j, i, fmt.format(val), ha='center', va='center',
                color=text_color, fontsize=9, fontweight='bold')