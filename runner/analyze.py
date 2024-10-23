import logging
import os
import shutil
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib.figure import Figure
from omegaconf import DictConfig

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
from retinal_rl.models.loss import ReconstructionLoss
from retinal_rl.models.objective import ContextT, Objective
from retinal_rl.util import FloatArray

logger = logging.getLogger(__name__)

init_dir = "initialization_analysis"


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
    if not cfg.use_wandb:
        _plot_and_save_histories(cfg, histories)

    cnn_analysis = cnn_statistics(device, test_set, brain, 1000)

    summary = brain.scan()
    filepath = os.path.join(cfg.system.run_dir, "brain_summary.txt")

    with open(filepath, "w") as f:
        f.write(summary)

    if cfg.use_wandb:
        wandb.save(filepath)

    if epoch == 0:
        _perform_initialization_analysis(cfg, brain, objective, train_set, cnn_analysis)

    _analyze_layers(cfg, cnn_analysis, epoch, copy_checkpoint)

    _perform_reconstruction_analysis(
        cfg, device, brain, objective, train_set, test_set, epoch, copy_checkpoint
    )


def _plot_and_save_histories(cfg: DictConfig, histories: Dict[str, List[float]]):
    hist_fig = plot_histories(histories)
    _save_figure(cfg, "", "histories", hist_fig)
    plt.close(hist_fig)


def _perform_initialization_analysis(
    cfg: DictConfig,
    brain: Brain,
    objective: Objective[ContextT],
    train_set: Imageset,
    cnn_analysis: Dict[str, Dict[str, FloatArray]],
):
    rf_sizes_fig = plot_receptive_field_sizes(cnn_analysis)
    _process_figure(cfg, False, rf_sizes_fig, init_dir, "receptive_field_sizes", 0)

    graph_fig = plot_brain_and_optimizers(brain, objective)
    _process_figure(cfg, False, graph_fig, init_dir, "brain_graph", 0)

    transforms = transform_base_images(train_set, num_steps=5, num_images=2)
    transforms_fig = plot_transforms(**transforms)
    _process_figure(cfg, False, transforms_fig, init_dir, "transforms", 0)


def _analyze_layers(
    cfg: DictConfig,
    cnn_analysis: Dict[str, Dict[str, FloatArray]],
    epoch: int,
    copy_checkpoint: bool,
):
    for layer_name, layer_data in cnn_analysis.items():
        if layer_name == "input":
            _analyze_input_layer(cfg, layer_data, epoch)
        else:
            _analyze_regular_layer(cfg, layer_name, layer_data, epoch, copy_checkpoint)


def _analyze_input_layer(
    cfg: DictConfig, layer_data: Dict[str, FloatArray], epoch: int
):
    if epoch == 0:
        layer_rfs = layer_receptive_field_plots(layer_data["receptive_fields"])
        _process_figure(cfg, False, layer_rfs, init_dir, "input_rfs", 0)

        num_channels = int(layer_data["num_channels"])
        for channel in range(num_channels):
            channel_fig = plot_channel_statistics(layer_data, "input", channel)
            _process_figure(
                cfg, False, channel_fig, init_dir, f"input_channel_{channel}", 0
            )


def _analyze_regular_layer(
    cfg: DictConfig,
    layer_name: str,
    layer_data: Dict[str, FloatArray],
    epoch: int,
    copy_checkpoint: bool,
):
    layer_rfs = layer_receptive_field_plots(layer_data["receptive_fields"])
    _process_figure(
        cfg, copy_checkpoint, layer_rfs, "receptive_fields", f"{layer_name}", epoch
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


def _perform_reconstruction_analysis(
    cfg: DictConfig,
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
        rec_dict = reconstruct_images(device, brain, decoder, train_set, test_set, 5)
        recon_fig = plot_reconstructions(**rec_dict, num_samples=5)
        _process_figure(
            cfg,
            copy_checkpoint,
            recon_fig,
            "reconstruction",
            f"{decoder}_reconstructions",
            epoch,
        )


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
