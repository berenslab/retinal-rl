import os
import shutil
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
from matplotlib.figure import Figure

from retinal_rl.classification.plot import (
    plot_input_distributions,
    plot_training_histories,
)
from retinal_rl.models.analysis.plot import plot_reconstructions, receptive_field_plots
from retinal_rl.models.analysis.statistics import (
    get_reconstructions,
    gradient_receptive_fields,
)
from retinal_rl.models.brain import Brain
import wandb

FigureDict = Dict[str, Figure | Dict[str, Any]]


def analyze(
    cfg: DictConfig,
    device: torch.device,
    brain: Brain,
    histories: Dict[str, List[float]],
    train_set: Dataset[Tuple[Tensor, int]],
    test_set: Dataset[Tuple[Tensor, int]],
    epoch: int,
    check_path: str = "",
):

    fig_dict: FigureDict = {}

    # Plot training histories
    if not cfg.logging.use_wandb:
        hist_fig = plot_training_histories(histories)
        fig_dict["Training_Histories"] = hist_fig

    # Plot input distributions if required
    if cfg.command.plot_inputs:
        rgb_fig = plot_input_distributions(train_set)
        fig_dict["Input_Distributions"] = rgb_fig

    # Plot receptive fields
    rf_dict = gradient_receptive_fields(device, brain.circuits["encoder"])
    fig_dict["Receptive_Fields"] = {}
    for lyr, rfs in rf_dict.items():
        rf_fig = receptive_field_plots(rfs)
        fig_dict["Receptive_Fields"][f"{lyr}_layer"] = rf_fig

    # Plot reconstructions
    rec_dict = get_reconstructions(device, brain, train_set, test_set, 5)
    recon_fig = plot_reconstructions(**rec_dict, num_samples=5)
    fig_dict["Reconstructions"] = recon_fig

    # Handle logging or saving of figures
    if cfg.logging.use_wandb:
        _log_figures(fig_dict, epoch)
    else:
        plot_path = cfg.system.plot_path

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        _save_figures(fig_dict, plot_path)

        if check_path:
            checkpoint_plot_path = os.path.join(check_path)
            if not os.path.exists(checkpoint_plot_path):
                os.makedirs(checkpoint_plot_path)
            shutil.copytree(plot_path, checkpoint_plot_path)


# Function to save figures from a dictionary recursively


def _log_figures(fig_dict: FigureDict, epoch) -> None:
    """Log figures to wandb."""
    wandb.log(fig_dict)  # , step=epoch)

    # Close the figures to free up memory
    for fig in fig_dict.values():
        if isinstance(fig, dict):
            for sub_fig in fig.values():
                plt.close(sub_fig)
        else:
            plt.close(fig)


def _save_figures(fig_dict: FigureDict, path: str):

    for key, fig in fig_dict.items():
        if isinstance(fig, dict):
            sub_dir = os.path.join(path, key)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            _save_figures(fig, sub_dir)
        else:
            fig_file = os.path.join(path, f"{key}.png")
            fig.savefig(fig_file)
            plt.close(fig)
