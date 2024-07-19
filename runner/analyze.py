import logging
import os
import shutil
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import wandb
from retinal_rl.classification.analysis.plot import (
    plot_image_distribution_analysis,
    plot_training_histories,
)
from retinal_rl.classification.analysis.statistics import (
    image_distribution_analysis,
    image_distribution_analysis_cnn,
)
from retinal_rl.models.analysis.plot import plot_reconstructions, receptive_field_plots
from retinal_rl.models.analysis.statistics import (
    get_reconstructions,
    gradient_receptive_fields,
)
from retinal_rl.models.brain import Brain
from retinal_rl.models.circuits.convolutional import ConvolutionalEncoder

FigureDict = Dict[str, Figure]

logger = logging.getLogger(__name__)


def analyze(
    cfg: DictConfig,
    device: torch.device,
    brain: Brain,
    histories: Dict[str, List[float]],
    train_set: Dataset[Tuple[Tensor, int]],
    test_set: Dataset[Tuple[Tensor, int]],
    epoch: int,
    copy_checkpoint: bool = False,
):
    fig_dict: FigureDict = {}

    testloader = DataLoader(test_set, batch_size=64, shuffle=False)

    # Plot training histories
    if not cfg.logging.use_wandb:
        hist_fig = plot_training_histories(histories)
        fig_dict["training-histories"] = hist_fig

    # Plot input distributions if required
    if cfg.logging.plot_inputs and epoch == 0:
        img_dict = image_distribution_analysis(device, testloader)
        img_fig = plot_image_distribution_analysis(img_dict)
        fig_dict["testset-analysis"] = img_fig

    # Plot receptive fields
    if "cnn_encoder" in brain.circuits:
        cnn_encoder = brain.circuits["cnn_encoder"]
        if isinstance(cnn_encoder, ConvolutionalEncoder):
            rf_dict = gradient_receptive_fields(device, cnn_encoder)
            for lyr, rfs in rf_dict.items():
                rf_fig = receptive_field_plots(rfs)
                fig_dict[f"receptive-fields/{lyr}-layer"] = rf_fig

            # CNN analysis
            cnn_analysis = image_distribution_analysis_cnn(device, test_set, cnn_encoder)
            for layer_name, layer_data in cnn_analysis.items():
                layer_fig = plot_image_distribution_analysis(layer_data)
                fig_dict[f"cnn-analysis/{layer_name}"] = layer_fig
        else:
            logger.warning(
                f"cnn_encoder is not a ConvolutionalEncoder, but a {type(cnn_encoder)}"
            )
    else:
        logger.info("cnn_encoder not found in brain circuits")
    rec_dict = get_reconstructions(device, brain, train_set, test_set, 5)
    recon_fig = plot_reconstructions(**rec_dict, num_samples=5)
    fig_dict["reconstructions"] = recon_fig

    # Handle logging or saving of figures
    if cfg.logging.use_wandb:
        _log_figures(fig_dict)
    else:
        _save_figures(cfg, epoch, fig_dict, copy_checkpoint)


def _wandb_title(title: str) -> str:
    # Split the title by slashes
    parts = title.split("/")

    def capitalize_part(part: str) -> str:
        # Split the part by dashes
        words = part.split("-")
        # Capitalize each word
        capitalized_words = [word.capitalize() for word in words]
        # Join the words with spaces
        return " ".join(capitalized_words)

    # Capitalize each part, then join with slashes
    capitalized_parts = [capitalize_part(part) for part in parts]
    return "/".join(capitalized_parts)


def _log_figures(fig_dict: FigureDict) -> None:
    """Log figures to wandb."""
    fig_dict_prefixed = {
        f"Figures/{_wandb_title(key)}": fig for key, fig in fig_dict.items()
    }
    wandb.log(fig_dict_prefixed, commit=False)

    # Close the figures to free up memory
    for fig in fig_dict.values():
        plt.close(fig)


def _save_figures(
    cfg: DictConfig, epoch: int, fig_dict: FigureDict, copy_checkpoint: bool
):
    plot_dir = cfg.system.plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    for key, fig in fig_dict.items():
        fig_dir = key.replace("/", os.sep)
        full_dir = os.path.join(plot_dir, fig_dir + ".png")
        os.makedirs(os.path.dirname(full_dir), exist_ok=True)
        fig.savefig(full_dir)
        plt.close(fig)

    if copy_checkpoint:
        checkpoint_plot_dir = f"{cfg.system.checkpoint_plot_dir}/checkpoint-epoch-{epoch}"
        os.makedirs(checkpoint_plot_dir, exist_ok=True)

        # Copy 'receptive-fields' directory
        src_dir = os.path.join(plot_dir, "receptive-fields")
        dst_dir = os.path.join(checkpoint_plot_dir, "receptive-fields")
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        # Copy 'reconstructions.png' file
        src_file = os.path.join(plot_dir, "reconstructions.png")
        dst_file = os.path.join(checkpoint_plot_dir, "reconstructions.png")
        if os.path.exists(src_file):
            os.makedirs(checkpoint_plot_dir, exist_ok=True)
            shutil.copy2(src_file, dst_file)
