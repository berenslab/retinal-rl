from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch import Tensor

def rescaleZeroOne(input):
    return (input - input.min()) / (input.max() - input.min())

def receptive_field_plots(lyr_rfs: NDArray[np.float64], max_cols: int = 8, rgb_rfs=False) -> Figure:
    """Plot the receptive fields of a convolutional layer."""
    ochns, nclrs, _, _ = lyr_rfs.shape
    rgb_rfs = rgb_rfs and (nclrs == 3)

    # Calculate the number of rows needed based on max_cols
    cols = min(ochns, max_cols)
    rows = (ochns // max_cols) * nclrs + (1 if ochns % max_cols > 0 else 0) * nclrs
    if rgb_rfs:
        rows = rows // 3

    fig, axs0 = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2, 1.6 * rows),
        squeeze=False,
    )

    axs = axs0.flat
    clrs = ["Red", "Green", "Blue"]
    cmaps = ["inferno", "viridis", "cividis"]

    for i in range(ochns):
        if not rgb_rfs:
            for j in range(nclrs):
                ax = axs[(i // max_cols) * nclrs * max_cols + (j * max_cols) + (i % max_cols)]
                im = ax.imshow(lyr_rfs[i, j, :, :], cmap=cmaps[j])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
                # Set title to channel i when j = 0
                if j == 0:
                    ax.set_title(f"Channel {i+1}")

                if i % max_cols == 0:
                    ax.set_ylabel(clrs[j])
                    fig.colorbar(im, ax=ax, cmap=cmaps[j], location="right")
                else:
                    fig.colorbar(im, ax=ax, cmap=cmaps[j], location="right")
        else:
            ax = axs[i]
            rescaled_img = rescaleZeroOne(lyr_rfs[i])
            im = ax.imshow(np.moveaxis(rescaled_img,0,2))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.set_title(f"Channel {i+1}")

    fig.tight_layout()  # Adjust layout to fit color bars
    return fig


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
