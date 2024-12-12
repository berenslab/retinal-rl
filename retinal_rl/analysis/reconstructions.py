
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from retinal_rl.analysis.plot import FigureLogger
from retinal_rl.classification.imageset import Imageset
from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ContextT, ReconstructionLoss
from retinal_rl.models.objective import Objective
from retinal_rl.util import FloatArray, NumpyEncoder


@dataclass
class Reconstructions:
    """Set of source images, inputs, and their reconstructions."""

    sources: list[tuple[FloatArray, int]]
    inputs: list[tuple[FloatArray, int]]
    estimates: list[tuple[FloatArray, int]]

@dataclass
class ReconstructionStatistics:
    """Results of image reconstruction for both training and test sets."""

    train: Reconstructions
    test: Reconstructions

# TODO: Make structure match the analyze / plot structure as receptive_fields

def perform_reconstruction_analysis(
    log: FigureLogger,
    analyses_dir: Path,
    device: torch.device,
    brain: Brain,
    objective: Objective[ContextT],
    train_set: Imageset,
    test_set: Imageset,
    epoch: int,
    copy_checkpoint: bool,
):
    reconstruction_decoders = [
        loss.target_decoder
        for loss in objective.losses
        if isinstance(loss, ReconstructionLoss)
    ]

    for decoder in reconstruction_decoders:
        norm_means, norm_stds = train_set.normalization_stats
        rec_dict = asdict(
            reconstruct_images(device, brain, decoder, train_set, test_set, 5)
        )
        # Save the reconstructions
        rec_path = analyses_dir / f"{decoder}_reconstructions_epoch_{epoch}.json"
        with open(rec_path, "w") as f:
            json.dump(rec_dict, f, cls=NumpyEncoder)

        recon_fig = plot_reconstructions(
            norm_means,
            norm_stds,
            *rec_dict["train"].values(),
            *rec_dict["test"].values(),
            num_samples=5,
        )
        log.log_figure(
            recon_fig,
            "reconstruction",
            f"{decoder}_reconstructions",
            epoch,
            copy_checkpoint,
        )


def reconstruct_images(
    device: torch.device,
    brain: Brain,
    decoder: str,
    test_set: Imageset,
    train_set: Imageset,
    sample_size: int,
) -> ReconstructionStatistics:
    """Compute reconstructions of a set of training and test images using a Brain model."""
    brain.eval()  # Set the model to evaluation mode

    def collect_reconstructions(
        imageset: Imageset, sample_size: int
    ) -> Reconstructions:
        """Collect reconstructions for a subset of a dataset."""
        source_subset: list[tuple[FloatArray, int]] = []
        input_subset: list[tuple[FloatArray, int]] = []
        estimates: list[tuple[FloatArray, int]] = []
        indices = torch.randperm(imageset.epoch_len())[:sample_size]

        with torch.no_grad():  # Disable gradient computation
            for index in indices:
                src, img, k = imageset[int(index)]
                src = src.to(device)
                img = img.to(device)
                stimulus = {"vision": img.unsqueeze(0)}
                response = brain(stimulus)
                rec_img = response[decoder].squeeze(0)
                pred_k = response["classifier"].argmax().item()
                source_subset.append((src.cpu().numpy(), k))
                input_subset.append((img.cpu().numpy(), k))
                estimates.append((rec_img.cpu().numpy(), pred_k))

        return Reconstructions(source_subset, input_subset, estimates)

    return ReconstructionStatistics(
        collect_reconstructions(train_set, sample_size),
        collect_reconstructions(test_set, sample_size),
    )


def plot_reconstructions(
    normalization_mean: list[float],
    normalization_std: list[float],
    train_sources: list[tuple[FloatArray, int]],
    train_inputs: list[tuple[FloatArray, int]],
    train_estimates: list[tuple[FloatArray, int]],
    test_sources: list[tuple[FloatArray, int]],
    test_inputs: list[tuple[FloatArray, int]],
    test_estimates: list[tuple[FloatArray, int]],
    num_samples: int,
) -> Figure:
    """Plot original and reconstructed images for both training and test sets, including the classes."""
    fig, axes = plt.subplots(6, num_samples, figsize=(15, 10))

    for i in range(num_samples):
        train_source, _ = train_sources[i]
        train_input, train_class = train_inputs[i]
        train_recon, train_pred = train_estimates[i]
        test_source, _ = test_sources[i]
        test_input, test_class = test_inputs[i]
        test_recon, test_pred = test_estimates[i]

        # Unnormalize the original images using the normalization lists
        # Arrays are already [C, H, W], need to move channels to last dimension
        train_source = (
            np.transpose(train_source, (1, 2, 0)) * normalization_std
            + normalization_mean
        )
        train_input = (
            np.transpose(train_input, (1, 2, 0)) * normalization_std
            + normalization_mean
        )
        train_recon = (
            np.transpose(train_recon, (1, 2, 0)) * normalization_std
            + normalization_mean
        )
        test_source = (
            np.transpose(test_source, (1, 2, 0)) * normalization_std
            + normalization_mean
        )
        test_input = (
            np.transpose(test_input, (1, 2, 0)) * normalization_std + normalization_mean
        )
        test_recon = (
            np.transpose(test_recon, (1, 2, 0)) * normalization_std + normalization_mean
        )

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
