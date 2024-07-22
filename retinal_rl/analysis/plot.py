from typing import Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from torch import Tensor

from retinal_rl.util import FloatArray


def plot_training_histories(histories: Dict[str, List[float]]) -> Figure:
    """Plot training and test losses over epochs.

    Args:
    ----
        histories (Dict[str, List[float]]): Dictionary containing training and test loss histories.

    Returns:
    -------
        Figure: Matplotlib figure containing the plotted histories.

    """
    fig: Figure
    axs: List[Axes]
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)  # type: ignore

    metrics: List[str] = ["total", "fraction_correct", "reconstruction"]
    for idx, metric in enumerate(metrics):
        ax: Axes = axs[idx]
        lbl: str = " ".join([word.capitalize() for word in metric.split("_")])
        ax.plot(  # type: ignore
            histories[f"train_{metric}"],
            label=f"{lbl} Training Error",
            color="black",
        )
        ax.plot(  # type: ignore
            histories[f"test_{metric}"],
            label=f"{lbl} Test Error",
            color="red",
        )
        ax.set_xlabel("Epochs")  # type: ignore
        ax.set_ylabel("Loss")  # type: ignore
        ax.legend()  # type: ignore
        ax.grid(True)  # type: ignore
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Training and Test Losses", fontsize=16)  # type: ignore
    return fig


def plot_channel_statistics(
    layer_data: Dict[str, FloatArray], layer_name: str, channel: int
) -> Figure:
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
    train_subset: List[Tuple[Tensor, int]],
    train_estimates: List[Tuple[Tensor, int]],
    test_subset: List[Tuple[Tensor, int]],
    test_estimates: List[Tuple[Tensor, int]],
    num_samples: int,
) -> Figure:
    """Plot original and reconstructed images for both training and test sets, including the classes.

    Args:
    ----
        train_subset (List[Tuple[Tensor, int]]): List of original training images and their classes.
        train_estimates (List[Tuple[Tensor, int]]): List of reconstructed training images and their predicted classes.
        test_subset (List[Tuple[Tensor, int]]): List of original test images and their classes.
        test_estimates (List[Tuple[Tensor, int]]): List of reconstructed test images and their predicted classes.
        num_samples (int): The number of samples to plot.

    Returns:
    -------
        Figure: The matplotlib Figure object with the plotted images.

    """
    fig, axes = plt.subplots(4, num_samples, figsize=(15, 10))

    for i in range(num_samples):
        # Unnormalize the original images
        train_original, train_class = train_subset[i]
        train_recon, train_pred = train_estimates[i]
        test_original, test_class = test_subset[i]
        test_recon, test_pred = test_estimates[i]

        train_original = train_original.permute(1, 2, 0).numpy() * 0.5 + 0.5
        train_recon = train_recon.permute(1, 2, 0).numpy() * 0.5 + 0.5
        test_original = test_original.permute(1, 2, 0).numpy() * 0.5 + 0.5
        test_recon = test_recon.permute(1, 2, 0).numpy() * 0.5 + 0.5

        axes[0, i].imshow(np.clip(train_original, 0, 1))
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Class: {train_class}")

        axes[1, i].imshow(np.clip(train_recon, 0, 1))
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Pred: {train_pred}")

        axes[2, i].imshow(np.clip(test_original, 0, 1))
        axes[2, i].axis("off")
        axes[2, i].set_title(f"Class: {test_class}")

        axes[3, i].imshow(np.clip(test_recon, 0, 1))
        axes[3, i].axis("off")
        axes[3, i].set_title(f"Pred: {test_pred}")

    # Set y-axis labels for each row

    fig.text(
        0.02,
        0.88,
        "Train Originals",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.02,
        0.62,
        "Train Reconstructions",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.02,
        0.38,
        "Test Originals",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.02,
        0.12,
        "Test Reconstructions",
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
