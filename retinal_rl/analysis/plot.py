"""Utility functions for plotting the results of statistical analyses."""

from typing import Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.fft as fft
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from torch import Tensor

from retinal_rl.models.brain import Brain
from retinal_rl.models.optimizer import BrainOptimizer, ContextT
from retinal_rl.util import FloatArray


def plot_brain_and_optimizers(
    brain: Brain, brain_optimizer: BrainOptimizer[ContextT]
) -> Figure:
    """Visualize the Brain's connectome organized by depth and highlight optimizer targets using border colors.

    Args:
    ----
    - brain: The Brain instance
    - brain_optimizer: The BrainOptimizer instance

    """
    graph = brain.connectome

    # Compute the depth of each node
    depths: Dict[str, int] = {}
    for node in nx.topological_sort(graph):
        depths[node] = max([depths[pred] for pred in graph.predecessors(node)] + [-1]) + 1

    # Create a position dictionary based on depth
    pos: Dict[str, Tuple[float, float]] = {}
    nodes_at_depth: Dict[int, List[str]] = {}
    for node, depth in depths.items():
        if depth not in nodes_at_depth:
            nodes_at_depth[depth] = []
        nodes_at_depth[depth].append(node)

    max_depth = max(depths.values())
    for depth, nodes in nodes_at_depth.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = ((i - width / 2) / (width + 1), -(max_depth - depth) / max_depth)

    # Set up the plot
    fig = plt.figure(figsize=(12, 10))

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color="gray", arrows=True)

    # Color scheme for different node types
    color_map = {"sensor": "lightblue", "circuit": "lightgreen"}

    # Generate colors for optimizers
    optimizer_colors = sns.color_palette("husl", len(brain_optimizer.optimizers))

    # Prepare node colors and edge colors
    node_colors: List[str] = []
    edge_colors: List[Tuple[float, float, float]] = []
    for node in graph.nodes():
        if node in brain.sensors:
            node_colors.append(color_map["sensor"])
        else:
            node_colors.append(color_map["circuit"])

        # Determine if the node is targeted by an optimizer
        edge_color = "none"
        for i, optimizer_name in enumerate(brain_optimizer.optimizers.keys()):
            if node in brain_optimizer.target_circuits[optimizer_name]:
                edge_color = optimizer_colors[i]
                break
        edge_colors.append(edge_color)

    # Draw nodes with a single call
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        edgecolors=edge_colors,
        node_size=4000,
        linewidths=5,
    )

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")

    # Add a legend for optimizers
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Optimizer: {name}",
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=15,
            markeredgewidth=3,
        )
        for name, color in zip(brain_optimizer.optimizers.keys(), optimizer_colors)
    ]

    # Add legend elements for sensor and circuit
    legend_elements.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Sensor",
                markerfacecolor=color_map["sensor"],
                markersize=15,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Circuit",
                markerfacecolor=color_map["circuit"],
                markersize=15,
            ),
        ]
    )

    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.title("Brain Connectome and Optimizer Targets")
    plt.tight_layout()
    plt.axis("off")

    return fig


def plot_receptive_field_sizes(results: Dict[str, Dict[str, FloatArray]]) -> Figure:
    """Plot the receptive field sizes for each layer of the convolutional part of the network.

    Args:
    ----
    - results: Dictionary containing the results from cnn_statistics function

    """
    # Get visual field size from the input shape
    input_shape = results["input"]["shape"]
    [_, height, width] = list(input_shape)

    # Calculate receptive field sizes for each layer
    rf_sizes: List[Tuple[int, int]] = []
    layer_names: List[str] = []
    for name, layer_data in results.items():
        if name == "input":
            continue
        rf = layer_data["receptive_fields"]
        rf_height, rf_width = rf.shape[2:]
        rf_sizes.append((rf_height, rf_width))
        layer_names.append(name)

    rf_sizes.reverse()
    layer_names.reverse()

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Reverse y-axis to match image coordinates
    ax.set_aspect("equal", "box")
    ax.set_title("Receptive Field Sizes of Convolutional Layers")

    # Set up grid lines
    ax.set_xticks(np.arange(0, width + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, height + 1, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    # Set up major ticks and labels
    major_ticks_x = [0, width // 2, width]
    major_ticks_y = [0, height // 2, height]
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticklabels(["0", f"{width // 2}", f"{width}"])
    ax.set_yticklabels([f"{height}", f"{height // 2}", "0"])

    # Use a color palette from seaborn
    colors = sns.color_palette("husl", n_colors=len(rf_sizes))

    # Plot receptive fields
    for (rf_height, rf_width), color, name in zip(rf_sizes, colors, layer_names):
        center_x, center_y = width // 2, height // 2
        rect = patches.Rectangle(
            (center_x - rf_width // 2, center_y - rf_height // 2),
            rf_width,
            rf_height,
            fill=True,
            facecolor=color,
            edgecolor=color,
            label=f"{name} ({rf_height}x{rf_width})",
        )
        ax.add_patch(rect)

    # Add legend
    ax.legend()

    plt.tight_layout()

    return fig


def plot_histories(histories: Dict[str, List[float]]) -> Figure:
    """Plot training and test losses over epochs.

    Args:
    ----
        histories (Dict[str, List[float]]): Dictionary containing training and test loss histories.

    Returns:
    -------
        Figure: Matplotlib figure containing the plotted histories.

    """
    train_metrics = [
        key.split("_", 1)[1] for key in histories.keys() if key.startswith("train_")
    ]
    test_metrics = [
        key.split("_", 1)[1] for key in histories.keys() if key.startswith("test_")
    ]

    # Use the intersection of train and test metrics to ensure we have both for each metric
    metrics = list(set(train_metrics) & set(test_metrics))

    # Determine the number of rows needed (2 metrics per row)
    num_rows = (len(metrics) + 1) // 2

    fig: Figure
    axs: List[Axes]
    fig, axs = plt.subplots(
        num_rows, 2, figsize=(15, 5 * num_rows), constrained_layout=True
    )
    axs = axs.flatten() if num_rows > 1 else [axs]

    for idx, metric in enumerate(metrics):
        ax: Axes = axs[idx]
        lbl: str = " ".join([word.capitalize() for word in metric.split("_")])

        if f"train_{metric}" in histories:
            ax.plot(
                histories[f"train_{metric}"],
                label=f"{lbl} Training",
                color="black",
            )
        if f"test_{metric}" in histories:
            ax.plot(
                histories[f"test_{metric}"],
                label=f"{lbl} Test",
                color="red",
            )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Value")
        ax.set_title(lbl)
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Remove any unused subplots
    for idx in range(len(metrics), len(axs)):
        fig.delaxes(axs[idx])

    fig.suptitle("Training and Test Metrics", fontsize=16)
    return fig


def plot_channel_statistics(
    layer_data: Dict[str, FloatArray], layer_name: str, channel: int
) -> Figure:
    """Plot receptive fields, pixel histograms, and autocorrelation plots for a single channel in a layer."""
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"Layer: {layer_name}, Channel: {channel}", fontsize=16)

    # Receptive Fields
    rf = layer_data["receptive_fields"][channel]
    _plot_receptive_fields(axs[0, 0], rf)
    axs[0, 0].set_title("Receptive Field")
    axs[0, 0].set_xlabel("X")
    axs[0, 0].set_ylabel("Y")

    # Pixel Histograms
    hist = layer_data["pixel_histograms"][channel]
    bin_edges = layer_data["histogram_bin_edges"]
    axs[1, 0].bar(
        bin_edges[:-1],
        hist,
        width=np.diff(bin_edges),
        align="edge",
        color="gray",
        edgecolor="black",
    )
    axs[1, 0].set_title("Channel Histogram")
    axs[1, 0].set_xlabel("Value")
    axs[1, 0].set_ylabel("Empirical Probability")

    # Autocorrelation plots
    # Plot average 2D autocorrelation and variance
    autocorr = fft.fftshift(layer_data["mean_autocorr"][channel])
    h, w = autocorr.shape
    extent = [-w // 2, w // 2, -h // 2, h // 2]
    im = axs[0, 1].imshow(
        autocorr, cmap="twilight", vmin=-1, vmax=1, origin="lower", extent=extent
    )
    axs[0, 1].set_title("Average 2D Autocorrelation")
    axs[0, 1].set_xlabel("Lag X")
    axs[0, 1].set_ylabel("Lag Y")
    fig.colorbar(im, ax=axs[0, 1])
    _set_integer_ticks(axs[0, 1])

    autocorr_sd = fft.fftshift(np.sqrt(layer_data["var_autocorr"][channel]))
    im = axs[0, 2].imshow(
        autocorr_sd, cmap="inferno", origin="lower", extent=extent, vmin=0
    )
    axs[0, 2].set_title("2D Autocorrelation SD")
    axs[0, 2].set_xlabel("Lag X")
    axs[0, 2].set_ylabel("Lag Y")
    fig.colorbar(im, ax=axs[0, 2])
    _set_integer_ticks(axs[0, 2])

    # Plot average 2D power spectrum
    log_power_spectrum = fft.fftshift(
        np.log1p(layer_data["mean_power_spectrum"][channel])
    )
    h, w = log_power_spectrum.shape

    im = axs[1, 1].imshow(
        log_power_spectrum, cmap="viridis", origin="lower", extent=extent, vmin=0
    )
    axs[1, 1].set_title("Average 2D Power Spectrum (log)")
    axs[1, 1].set_xlabel("Frequency X")
    axs[1, 1].set_ylabel("Frequency Y")
    fig.colorbar(im, ax=axs[1, 1])
    _set_integer_ticks(axs[1, 1])

    log_power_spectrum_sd = fft.fftshift(
        np.log1p(np.sqrt(layer_data["var_power_spectrum"][channel]))
    )
    im = axs[1, 2].imshow(
        log_power_spectrum_sd,
        cmap="viridis",
        origin="lower",
        extent=extent,
        vmin=0,
    )
    axs[1, 2].set_title("2D Power Spectrum SD")
    axs[1, 2].set_xlabel("Frequency X")
    axs[1, 2].set_ylabel("Frequency Y")
    fig.colorbar(im, ax=axs[1, 2])
    _set_integer_ticks(axs[1, 2])

    plt.tight_layout()
    return fig


def _set_integer_ticks(ax: Axes):
    """Set integer ticks for both x and y axes."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


# Function to plot the original and reconstructed images
def plot_reconstructions(
    train_sources: List[Tuple[Tensor, int]],
    train_inputs: List[Tuple[Tensor, int]],
    train_estimates: List[Tuple[Tensor, int]],
    test_sources: List[Tuple[Tensor, int]],
    test_inputs: List[Tuple[Tensor, int]],
    test_estimates: List[Tuple[Tensor, int]],
    num_samples: int,
) -> Figure:
    """Plot original and reconstructed images for both training and test sets, including the classes.

    Args:
    ----
        train_source (List[Tuple[Tensor, int]]): List of original source images and their classes.
        train_input (List[Tuple[Tensor, int]]): List of original training images and their classes.
        train_estimates (List[Tuple[Tensor, int]]): List of reconstructed training images and their predicted classes.
        test_source (List[Tuple[Tensor, int]]): List of original source images and their classes.
        test_input (List[Tuple[Tensor, int]]): List of original test images and their classes.
        test_estimates (List[Tuple[Tensor, int]]): List of reconstructed test images and their predicted classes.
        num_samples (int): The number of samples to plot.

    Returns:
    -------
        Figure: The matplotlib Figure object with the plotted images.

    """
    fig, axes = plt.subplots(6, num_samples, figsize=(15, 10))

    for i in range(num_samples):
        train_source, _ = train_sources[i]
        train_input, train_class = train_inputs[i]
        train_recon, train_pred = train_estimates[i]
        test_source, _ = test_sources[i]
        test_input, test_class = test_inputs[i]
        test_recon, test_pred = test_estimates[i]

        # Unnormalize the original images
        train_source = train_source.permute(1, 2, 0).numpy() * 0.5 + 0.5
        train_input = train_input.permute(1, 2, 0).numpy() * 0.5 + 0.5
        train_recon = train_recon.permute(1, 2, 0).numpy() * 0.5 + 0.5
        test_source = test_source.permute(1, 2, 0).numpy() * 0.5 + 0.5
        test_input = test_input.permute(1, 2, 0).numpy() * 0.5 + 0.5
        test_recon = test_recon.permute(1, 2, 0).numpy() * 0.5 + 0.5

        axes[0, i].imshow(np.clip(train_source, 0, 1))
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Class: {train_class}")

        axes[1, i].imshow(np.clip(train_input, 0, 1))
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Class: {train_class}")

        axes[2, i].imshow(np.clip(train_recon, 0, 1))
        axes[2, i].axis("off")
        axes[2, i].set_title(f"Pred: {train_pred}")

        axes[3, i].imshow(np.clip(test_source, 0, 1))
        axes[3, i].axis("off")
        axes[3, i].set_title(f"Class: {test_class}")

        axes[4, i].imshow(np.clip(test_input, 0, 1))
        axes[4, i].axis("off")
        axes[4, i].set_title(f"Class: {test_class}")

        axes[5, i].imshow(np.clip(test_recon, 0, 1))
        axes[5, i].axis("off")
        axes[5, i].set_title(f"Pred: {test_pred}")

    # Set y-axis labels for each row

    fig.text(
        0.02,
        0.90,
        "Train Source",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.02,
        0.74,
        "Train Input",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.02,
        0.56,
        "Train Recon.",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.02,
        0.40,
        "Test Source",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.02,
        0.24,
        "Test Input",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.02,
        0.08,
        "Test Recon.",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
    )

    plt.tight_layout()
    return fig


def _plot_receptive_fields(ax: Axes, rf: FloatArray):
    """Plot full-color receptive field and individual color channels for CIFAR-10 range (-1 to 1)."""
    # Clear the main axes
    ax.clear()
    ax.axis("off")

    # Create a GridSpec within the given axes
    gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())

    rf_full = np.moveaxis(rf, 0, -1)  # Move channel axis to the last dimension
    rf_min = rf_full.min()
    rf_max = rf_full.max()
    rf_full = (rf_full - rf_min) / (rf_max - rf_min)
    # Full-color receptive field

    ax_full = ax.figure.add_subplot(gs[0, 0])
    ax_full.imshow(rf_full)
    ax_full.set_title("Full Color")
    ax_full.axis("off")

    # Individual color channels
    channels = ["Red", "Green", "Blue"]
    cmaps = ["RdGy_r", "RdYlGn", "PuOr"]  # Diverging colormaps centered at 0
    positions = [(0, 1), (1, 0), (1, 1)]  # Correct positions for a 2x2 grid
    for i in range(3):
        row, col = positions[i]
        ax_channel = ax.figure.add_subplot(gs[row, col])
        im = ax_channel.imshow(rf[i], cmap=cmaps[i], vmin=rf_min, vmax=rf_max)
        ax_channel.set_title(channels[i])
        ax_channel.axis("off")
        plt.colorbar(im, ax=ax_channel, fraction=0.046, pad=0.04)

    # Add min and max values as text
    ax.text(
        0.5,
        -0.05,
        f"Min: {rf.min():.2f}, Max: {rf.max():.2f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )


def layer_receptive_field_plots(lyr_rfs: FloatArray, max_cols: int = 8) -> Figure:
    """Plot the receptive fields of a convolutional layer."""
    ochns, _, _, _ = lyr_rfs.shape

    # Calculate the number of rows needed based on max_cols
    cols = min(ochns, max_cols)
    rows = ochns // cols + (1 if ochns % cols > 0 else 0)

    fig, axs0 = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2, 1.6 * rows),
        squeeze=False,
    )

    axs = axs0.flat

    for i in range(ochns):
        ax = axs[i]
        data = np.moveaxis(lyr_rfs[i], 0, -1)  # Move channel axis to the last dimension
        data_min = data.min()
        data_max = data.max()
        data = (data - data_min) / (data_max - data_min)
        ax.imshow(data)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_title(f"Channel {i+1}")

    fig.tight_layout()  # Adjust layout to fit color bars
    return fig
