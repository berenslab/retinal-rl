from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray


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


def plot_image_distribution_analysis(
    analysis_data: Dict[str, NDArray[np.float64]],
) -> Figure:
    """Create a multi-panel plot for each channel showing pixel intensity histogram and power spectrum projections.

    Args:
    ----
    analysis_data: Dict[str, NDArray[np.float64]]
        The output from image_distribution_analysis function.

    Returns:
    -------
    Figure: A matplotlib Figure object containing the plots.

    """
    num_channels = analysis_data["channel_histograms"].shape[0]

    # Create a figure with two subplots per channel
    fig, axs = plt.subplots(
        num_channels, 2, figsize=(16, 6 * num_channels), squeeze=False
    )
    fig.suptitle("Image Distribution Analysis by Channel", fontsize=16)

    bin_edges = analysis_data["bin_edges"]

    for i in range(num_channels):
        # Plot histogram
        ax_hist = axs[i, 0]
        channel_hist = analysis_data["channel_histograms"][i]
        ax_hist.plot(bin_edges[:-1], channel_hist, linewidth=2, color="black")
        ax_hist.fill_between(bin_edges[:-1], channel_hist, alpha=0.3, color="black")
        ax_hist.set_xlabel("Pixel/Activation Value")
        ax_hist.set_ylabel("Normalized Frequency")
        ax_hist.set_title(f"Channel {i} - Pixel Intensity Histogram")
        ax_hist.grid(True, linestyle="--", alpha=0.7)

        # Plot non-redundant half
        ax_spec = axs[i, 1]
        x_len = len(analysis_data["x_mean_log_power_spectrum"][i]) // 2 + 1
        x_freqs = range(x_len)
        x_means = analysis_data["x_mean_log_power_spectrum"][i][:x_len]
        x_sds = analysis_data["x_sd_log_power_spectrum"][i][:x_len]
        y_len = len(analysis_data["y_mean_log_power_spectrum"][i]) // 2 + 1
        y_freqs = range(y_len)
        y_means = analysis_data["y_mean_log_power_spectrum"][i][:y_len]
        y_sds = analysis_data["y_sd_log_power_spectrum"][i][:y_len]
        ax_spec.set_xlim(0, x_len)
        ax_spec.set_ylim(0, y_len)

        # X-axis projection
        ax_spec.plot(
            x_freqs,
            x_means,
            label="X-axis Mean",
            color="blue",
        )
        ax_spec.fill_between(
            x_freqs,
            x_means - x_sds,
            x_means + x_sds,
            alpha=0.3,
            color="blue",
        )

        # Y-axis projection
        ax_spec.plot(
            y_freqs,
            y_means,
            label="Y-axis Mean",
            color="red",
        )
        ax_spec.fill_between(
            y_freqs,
            y_means - y_sds,
            y_means + y_sds,
            alpha=0.3,
            color="red",
        )

        ax_spec.set_xlabel("Frequency")
        ax_spec.set_ylabel("Log Power")
        ax_spec.set_title(f"Channel {i} - Power Spectrum Projections")
        ax_spec.legend()
        ax_spec.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the suptitle

    return fig
