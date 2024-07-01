import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

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


def analyze(
    cfg: DictConfig,
    brain: Brain,
    histories: Dict[str, List[float]],
    train_set: Dataset[Tuple[Tensor, int]],
    test_set: Dataset[Tuple[Tensor, int]],
    device: torch.device,
):
    hist_fig = plot_training_histories(histories)
    hist_path = os.path.join(cfg.system.plot_path, "histories")
    hist_fig.savefig(hist_path)
    plt.close()

    if cfg.command.plot_inputs:
        rgb_fig = plot_input_distributions(train_set)
        rgb_path = os.path.join(cfg.system.plot_path, "input-distributions")
        rgb_fig.savefig(rgb_path)
        plt.close()

    checkpoint_analyze(
        cfg.system.plot_path,
        brain,
        train_set,
        test_set,
        device,
    )


def checkpoint_analyze(
    checkpoint_plot_path: str,
    brain: Brain,
    train_set: Dataset[Tuple[Tensor, int]],
    test_set: Dataset[Tuple[Tensor, int]],
    device: torch.device,
):
    # check for the existence of the checkpoint_plot_path

    if not os.path.exists(checkpoint_plot_path):
        os.makedirs(checkpoint_plot_path)
    rf_sub_path = os.path.join(checkpoint_plot_path, "receptive-fields")
    if not os.path.exists(rf_sub_path):
        os.makedirs(rf_sub_path)

    rf_dict = gradient_receptive_fields(device, brain.circuits["encoder"])
    for lyr, rfs in rf_dict.items():
        rf_fig = receptive_field_plots(rfs)
        rf_path = os.path.join(rf_sub_path, f"{lyr}-layer-receptive-fields")
        rf_fig.savefig(rf_path)
        plt.close()

    rec_dict = get_reconstructions(device, brain, train_set, test_set, 5)
    recon_fig = plot_reconstructions(**rec_dict, num_samples=5)
    recon_path = os.path.join(checkpoint_plot_path, "reconstructions")
    recon_fig.savefig(recon_path)
    plt.close()
