from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset


def plot_training_histories(histories: Dict[str, List[float]]) -> Figure:
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    metrics = ["total", "fraction_correct", "reconstruction"]
    for idx, metric in enumerate(metrics):
        ax = axs[idx]
        # split at underscores and capitalize each word
        lbl = " ".join([word.capitalize() for word in metric.split("_")])
        ax.plot(
            histories[f"train_{metric}"],
            label=f"{lbl} Training Error",
            color="black",
        )
        ax.plot(
            histories[f"test_{metric}"],
            label=f"{lbl} Test Error",
            color="red",
        )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        # Force integer x labels
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Training and Test Losses", fontsize=16)

    return fig


def plot_input_distributions(dataset: Dataset[Tuple[Tensor, int]]) -> Figure:
    # Find min and max pixel values
    min_val = float("inf")
    max_val = float("-inf")

    for img, _ in dataset:
        min_val = min(min_val, img.min().item())
        max_val = max(max_val, img.max().item())

    print(f"Min pixel value: {min_val}")
    print(f"Max pixel value: {max_val}")

    # Initialize histogram bins
    bins = 40
    hist_bins: NDArray[np.float64] = np.linspace(min_val, max_val, bins + 1)
    hist_red = np.zeros(bins)
    hist_green = np.zeros(bins)
    hist_blue = np.zeros(bins)

    # Collect pixel values for each channel in a more memory-efficient way
    for img, _ in dataset:
        hist_red += np.histogram(img[0].flatten(), bins=hist_bins)[0]
        hist_green += np.histogram(img[1].flatten(), bins=hist_bins)[0]
        hist_blue += np.histogram(img[2].flatten(), bins=hist_bins)[0]

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["r", "g", "b"]
    titles = ["Red Channel", "Green Channel", "Blue Channel"]

    histograms = [hist_red, hist_green, hist_blue]
    for i, ax in enumerate(axes):
        ax.bar(
            hist_bins[:-1],
            histograms[i],
            width=np.diff(hist_bins),
            color=colors[i],
            alpha=0.7,
            align="edge",
        )
        ax.set_title(titles[i])
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()

    return fig
