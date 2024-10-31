import json
import logging
import os
import shutil
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
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
from retinal_rl.classification.imageset import Imageset
from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ReconstructionLoss
from retinal_rl.models.objective import ContextT, Objective
from retinal_rl.util import FloatArray

logger = logging.getLogger(__name__)

init_dir = "initialization_analysis"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_statistics(cfg: DictConfig, stats: Dict[str, Any], epoch: int) -> None:
    """Save statistics to a JSON file in the plot directory."""
    stats_dir = os.path.join(cfg.path.plot_dir, "statistics")
    os.makedirs(stats_dir, exist_ok=True)

    filename = os.path.join(stats_dir, f"epoch_{epoch}_stats.json")
    with open(filename, "w") as f:
        json.dump(stats, f, cls=NumpyEncoder)

    if cfg.logging.use_wandb:
        wandb.save(filename, base_path=cfg.path.plot_dir, policy="now")


def collect_statistics(
    cfg: DictConfig,
    device: torch.device,
    brain: Brain,
    objective: Objective[ContextT],
    train_set: Imageset,
    test_set: Imageset,
    epoch: int,
) -> Dict[str, Any]:
    """Collect all statistics without plotting."""
    stats = {}

    # Collect CNN statistics
    cnn_stats = cnn_statistics(
        device,
        test_set,
        brain,
        cfg.logging.channel_analysis,
        cfg.logging.plot_sample_size,
    )
    stats["cnn_analysis"] = cnn_stats

    # Collect reconstruction statistics if applicable
    reconstruction_decoders = [
        loss.target_decoder
        for loss in objective.losses
        if isinstance(loss, ReconstructionLoss)
    ]

    if reconstruction_decoders:
        rec_stats = {}
        for decoder in reconstruction_decoders:
            rec_dict = reconstruct_images(
                device, brain, decoder, train_set, test_set, 5
            )
            rec_stats[str(decoder)] = rec_dict
        stats["reconstruction_analysis"] = rec_stats

    # If it's initialization epoch, collect additional stats
    if epoch == 0:
        # Save brain summary
        stats["brain_summary"] = brain.scan()

        # Save transform statistics
        transforms = transform_base_images(train_set, num_steps=5, num_images=2)
        stats["transforms"] = transforms

    return stats


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
    # First collect all statistics
    stats = collect_statistics(
        cfg, device, brain, objective, train_set, test_set, epoch
    )

    # Save statistics to file
    save_statistics(cfg, stats, epoch)

    # Plot histories if not using wandb
    if not cfg.logging.use_wandb:
        _plot_and_save_histories(cfg, histories)

    # Plot CNN analysis
    cnn_analysis = stats["cnn_analysis"]

    if epoch == 0:
        _plot_initialization_analysis(cfg, brain, objective, train_set, cnn_analysis)

    _plot_layers(cfg, cnn_analysis, epoch, copy_checkpoint)

    # Plot reconstruction analysis if available
    if "reconstruction_analysis" in stats:
        _plot_reconstruction_analysis(
            cfg, train_set, stats["reconstruction_analysis"], epoch, copy_checkpoint
        )


def _plot_initialization_analysis(
    cfg: DictConfig,
    brain: Brain,
    objective: Objective[ContextT],
    train_set: Imageset,
    cnn_analysis: Dict[str, Dict[str, FloatArray]],
):
    # Save brain summary to file
    filepath = os.path.join(cfg.path.run_dir, "brain_summary.txt")
    with open(filepath, "w") as f:
        f.write(brain.scan())

    if cfg.logging.use_wandb:
        wandb.save(filepath, base_path=cfg.path.run_dir, policy="now")

    # Plot various initialization analyses
    rf_sizes_fig = plot_receptive_field_sizes(cnn_analysis)
    _process_figure(cfg, False, rf_sizes_fig, init_dir, "receptive_field_sizes", 0)

    graph_fig = plot_brain_and_optimizers(brain, objective)
    _process_figure(cfg, False, graph_fig, init_dir, "brain_graph", 0)

    transforms = transform_base_images(train_set, num_steps=5, num_images=2)
    transforms_fig = plot_transforms(**transforms)
    _process_figure(cfg, False, transforms_fig, init_dir, "transforms", 0)

    _plot_input_layer(cfg, cnn_analysis["input"], cfg.logging.channel_analysis)


def _plot_reconstruction_analysis(
    cfg: DictConfig,
    train_set: Imageset,
    rec_stats: Dict[str, Dict],
    epoch: int,
    copy_checkpoint: bool,
):
    norm_means, norm_stds = train_set.normalization_stats
    for decoder, rec_dict in rec_stats.items():
        recon_fig = plot_reconstructions(
            norm_means, norm_stds, **rec_dict, num_samples=5
        )
        _process_figure(
            cfg,
            copy_checkpoint,
            recon_fig,
            "reconstruction",
            f"{decoder}_reconstructions",
            epoch,
        )


# Rest of the helper functions remain the same
def _plot_and_save_histories(cfg: DictConfig, histories: Dict[str, List[float]]):
    hist_fig = plot_histories(histories)
    _save_figure(cfg, "", "histories", hist_fig)
    plt.close(hist_fig)


def _plot_layers(
    cfg: DictConfig,
    cnn_analysis: Dict[str, Dict[str, FloatArray]],
    epoch: int,
    copy_checkpoint: bool,
):
    for layer_name, layer_data in cnn_analysis.items():
        if layer_name != "input":
            _plot_regular_layer(
                cfg,
                layer_name,
                layer_data,
                epoch,
                copy_checkpoint,
                cfg.logging.channel_analysis,
            )


def _plot_input_layer(
    cfg: DictConfig,
    layer_data: Dict[str, FloatArray],
    channel_analysis: bool,
):
    layer_rfs = layer_receptive_field_plots(layer_data["receptive_fields"])
    _process_figure(cfg, False, layer_rfs, init_dir, "input_rfs", 0)

    if channel_analysis:
        num_channels = int(layer_data["num_channels"])
        for channel in range(num_channels):
            channel_fig = plot_channel_statistics(layer_data, "input", channel)
            _process_figure(
                cfg, False, channel_fig, init_dir, f"input_channel_{channel}", 0
            )


def _plot_regular_layer(
    cfg: DictConfig,
    layer_name: str,
    layer_data: Dict[str, FloatArray],
    epoch: int,
    copy_checkpoint: bool,
    channel_analysis: bool,
):
    layer_rfs = layer_receptive_field_plots(layer_data["receptive_fields"])
    _process_figure(
        cfg, copy_checkpoint, layer_rfs, "receptive_fields", f"{layer_name}", epoch
    )

    if channel_analysis:
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


def _save_figure(cfg: DictConfig, sub_dir: str, file_name: str, fig: Figure) -> None:
    dir = os.path.join(cfg.path.plot_dir, sub_dir)
    os.makedirs(dir, exist_ok=True)
    file_name = os.path.join(dir, f"{file_name}.png")
    fig.savefig(file_name)


def _checkpoint_copy(cfg: DictConfig, sub_dir: str, file_name: str, epoch: int) -> None:
    src_path = os.path.join(cfg.path.plot_dir, sub_dir, f"{file_name}.png")

    dest_dir = os.path.join(cfg.path.checkpoint_plot_dir, f"epoch_{epoch}", sub_dir)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"{file_name}.png")

    shutil.copy2(src_path, dest_path)


def _wandb_title(title: str) -> str:
    parts = title.split("/")

    def capitalize_part(part: str) -> str:
        words = part.split("_")
        capitalized_words = [word.capitalize() for word in words]
        return " ".join(capitalized_words)

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
    if cfg.logging.use_wandb:
        title = f"{_wandb_title(sub_dir)}/{_wandb_title(file_name)}"
        img = wandb.Image(fig)
        wandb.log({title: img}, commit=False)
    else:
        _save_figure(cfg, sub_dir, file_name, fig)
        if copy_checkpoint:
            _checkpoint_copy(cfg, sub_dir, file_name, epoch)
    plt.close(fig)
