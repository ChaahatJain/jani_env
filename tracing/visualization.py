""" Simple code for plotting heatmaps. 
    Code from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html 
"""
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch 

def visualize_hooked_model(hooked_model):
    for k,v in hooked_model.layers.items():    
        logging.info(f"Visualizing layer: {k}")
        visualize_single_pass(v.layer_forward, f"Forward pass in layer {k}")
        visualize_single_pass(v.layer_backward, f"Backward pass in layer {k}")
        accumulation_method = "mean"
        visualize_both_passes_fn(v.layer_forward, v.layer_backward, accumulation_method, f"{accumulation_method} forward and backward passes in layer {k}")

def visualize_single_pass(single_pass, label_name=None):
    # Visualizes a single forward or backward pass with n instances.
    neuron_idx = [x for x in range(len(single_pass[0]))]
    instance_idx = [x for x in range(len(single_pass))]
    if label_name is None:
        label_name = "values"
        
    layer_values = np.stack([x.detach().numpy() for x in single_pass])
    fig, ax = plt.subplots()

    im, cbar = heatmap(layer_values, instance_idx, neuron_idx, ax=ax,
                       cmap="YlGn", title=label_name)
    texts = annotate_heatmap(im, valfmt="{x:.4f}")

    fig.tight_layout()
    plt.show()

def visualize_both_passes_fn(forward_passes, backward_passes, accumulation="mean", label_name=None):    
    try:
        assert(len(forward_passes) == len(backward_passes))
    except AssertionError:
        logging.info("WARNING! Number of forward passes and backward passes is not equal")

    try:
        assert(len(forward_passes[0]) == len(backward_passes[0]))
    except AssertionError:
        logging.info("ERROR! Number of forward and backward neurons is different!")
        raise Exception("Please ensure the number of neurons match.")
         
    neuron_idx = [x for x in range(len(forward_passes[0]))]
    forward_np = [x.detach().numpy() for x in forward_passes]
    backward_np = [x.detach().numpy() for x in backward_passes]
    
    # Default is accumulation
    if accumulation == "std":
        forward_accum = np.std(forward_np, axis=0)
        backward_accum = np.std(backward_np, axis=0)
        x_labels = ["std_forward", "std_backward"]
    else:
        forward_accum = np.mean(forward_np, axis=0) 
        backward_accum = np.mean(backward_np, axis=0)
        x_labels = ["mean_forward", "mean_backward"]

    fig, ax = plt.subplots()

    im, cbar = heatmap(np.array([forward_accum, backward_accum]), x_labels, neuron_idx, ax=ax,
                       cmap="YlGn", title=label_name)
    texts = annotate_heatmap(im, valfmt="{x:.5f}")

    fig.tight_layout()
    plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", title="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
