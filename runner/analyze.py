import logging
import os
import shutil
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig

import wandb
from retinal_rl.analysis.plot import (
    layer_receptive_field_plots,
    plot_brain_and_optimizers,
    plot_channel_statistics,
    plot_histories,
    plot_receptive_field_sizes,
    plot_reconstructions,
    plot_transforms,
)
from retinal_rl.analysis.statistics import (
    cnn_statistics,
    reconstruct_images,
    transform_base_images,
)
from retinal_rl.dataset import Imageset
from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import ContextT, Objective

logger = logging.getLogger(__name__)

init_dir = "initialization_analysis"


def _save_figure(cfg: DictConfig, sub_dir: str, file_name: str, fig: Figure) -> None:
    dir = os.path.join(cfg.system.plot_dir, sub_dir)
    os.makedirs(dir, exist_ok=True)
    file_name = os.path.join(dir, f"{file_name}.png")
    fig.savefig(file_name)


def _checkpoint_copy(cfg: DictConfig, sub_dir: str, file_name: str, epoch: int) -> None:
    src_path = os.path.join(cfg.system.plot_dir, sub_dir, f"{file_name}.png")

    dest_dir = os.path.join(
        cfg.system.checkpoint_plot_dir, "checkpoints", f"epoch_{epoch}", sub_dir
    )
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"{file_name}.png")

    shutil.copy2(src_path, dest_path)


def _wandb_title(title: str) -> str:
    # Split the title by slashes
    parts = title.split("/")

    def capitalize_part(part: str) -> str:
        # Split the part by dashes
        words = part.split("_")
        # Capitalize each word
        capitalized_words = [word.capitalize() for word in words]
        # Join the words with spaces
        return " ".join(capitalized_words)

    # Capitalize each part, then join with slashes
    capitalized_parts = [capitalize_part(part) for part in parts]
    return "/".join(capitalized_parts)


def _process_figure(
    cfg: DictConfig,
    copy_checkpoint: bool,
    fig: Figure,
    sub_dir: str,
    file_name: str,
    epoch: int,
) -> None:
    if cfg.use_wandb:
        title = f"{_wandb_title(sub_dir)}/{_wandb_title(file_name)}"
        img = wandb.Image(fig)
        wandb.log({title: img}, commit=False)
    else:
        _save_figure(cfg, sub_dir, file_name, fig)
        if copy_checkpoint:
            _checkpoint_copy(cfg, sub_dir, file_name, epoch)
    plt.close(fig)


def analyze(
    cfg: DictConfig,
    device: torch.device,
    brain: Brain,
    objective: Objective[ContextT],
    histories: Dict[str, List[float]],
    train_set: Imageset,
    test_set: Imageset,
    epoch: int,
    copy_checkpoint: bool = False,
):
    # Plot training histories (this never gets logged to wandb)
    if not cfg.use_wandb:
        hist_fig = plot_histories(histories)
        _save_figure(cfg, "", "histories", hist_fig)
        plt.close(hist_fig)

    # CNN analysis
    cnn_analysis = cnn_statistics(device, test_set, brain, 1000)

    # Initialization analysis
    if epoch == 0:
        rf_sizes_fig = plot_receptive_field_sizes(cnn_analysis)
        _process_figure(cfg, False, rf_sizes_fig, init_dir, "receptive_field_sizes", 0)
        graph_fig = plot_brain_and_optimizers(brain, objective)
        _process_figure(cfg, False, graph_fig, init_dir, "brain_graph", 0)
        transforms = transform_base_images(train_set, num_steps=5, num_images=2)
        transforms_fig = plot_transforms(**transforms)
        _process_figure(cfg, False, transforms_fig, init_dir, "transforms", 0)

    for layer_name, layer_data in cnn_analysis.items():
        if layer_name == "input":
            if epoch == 0:
                layer_rfs = layer_receptive_field_plots(layer_data["receptive_fields"])
                _process_figure(cfg, False, layer_rfs, init_dir, "input_rfs", 0)

                num_channels = int(layer_data["num_channels"])
                for channel in range(num_channels):
                    channel_fig = plot_channel_statistics(layer_data, layer_name, channel)
                    _process_figure(
                        cfg,
                        False,
                        channel_fig,
                        init_dir,
                        f"input_channel_{channel}",
                        0,
                    )

            continue

        layer_rfs = layer_receptive_field_plots(layer_data["receptive_fields"])
        _process_figure(
            cfg,
            copy_checkpoint,
            layer_rfs,
            "receptive_fields",
            f"{layer_name}",
            epoch,
        )

        num_channels = int(layer_data["num_channels"])
        for channel in range(num_channels):
            channel_fig = plot_channel_statistics(layer_data, layer_name, channel)
            _process_figure(
                cfg,
                copy_checkpoint,
                channel_fig,
                f"{layer_name}_layer_channel_analysis",
                f"channel_{channel}",
                epoch,
            )
    rec_dict = reconstruct_images(device, brain, train_set, test_set, 5)
    recon_fig = plot_reconstructions(**rec_dict, num_samples=5)
    _process_figure(
        cfg, copy_checkpoint, recon_fig, "reconstruction", "reconstructions", epoch
    )
