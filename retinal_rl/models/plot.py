from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch import Tensor


def receptive_field_plots(lyr_rfs: NDArray[np.float64]) -> Figure:
    """Plot the receptive fields of a convolutional layer."""
    ochns, nclrs, _, _ = lyr_rfs.shape

    fig, axs0 = plt.subplots(
        nclrs,
        ochns,
        figsize=(ochns * 1.5, nclrs),
    )

    axs = axs0.flat
    clrs = ["Red", "Green", "Blue"]
    cmaps = ["inferno", "viridis", "cividis"]

    for i in range(ochns):
        mx = np.amax(lyr_rfs[i])
        mn = np.amin(lyr_rfs[i])

        for j in range(nclrs):
            ax = axs[i + ochns * j]
            # hght,wdth = lyr_rfs[i,j,:,:].shape
            im = ax.imshow(lyr_rfs[i, j, :, :], cmap=cmaps[j], vmin=mn, vmax=mx)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

            if i == 0:
                fig.colorbar(im, ax=ax, cmap=cmaps[j], label=clrs[j], location="left")
            else:
                fig.colorbar(im, ax=ax, cmap=cmaps[j], location="left")

    return fig


# Function to plot the original and reconstructed images
def plot_reconstructions(
    originals: List[Tensor], reconstructions: List[Tensor], num_samples: int
) -> Figure:
    """Plot original and reconstructed images.

    Args:
    ----
        originals (List[Tensor]): List of original images.
        reconstructions (List[Tensor]): List of reconstructed images.
        num_samples (int): The number of samples to plot.

    Returns:
    -------
        Figure: The matplotlib Figure object with the plotted images.

    """
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))

    for i in range(num_samples):
        # Unnormalize the original images
        original = originals[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        recon = reconstructions[i].permute(1, 2, 0).numpy() * 0.5 + 0.5

        axes[0, i].imshow(np.clip(original, 0, 1))
        axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(recon, 0, 1))
        axes[1, i].axis("off")

    axes[0, 0].set_title("Originals")
    axes[1, 0].set_title("Reconstructions")

    plt.tight_layout()
    return fig
