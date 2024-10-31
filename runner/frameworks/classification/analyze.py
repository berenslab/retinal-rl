import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path
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
    CNNStatistics,
    LayerStatistics,
    cnn_statistics,
    reconstruct_images,
    transform_base_images,
)
from retinal_rl.classification.imageset import Imageset
from retinal_rl.models.brain import Brain
from retinal_rl.models.loss import ReconstructionLoss
from retinal_rl.models.objective import ContextT, Objective

### Infrastructure ###


logger = logging.getLogger(__name__)

init_dir = "initialization_analysis"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


### Analysis ###


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
    results_dir = Path(cfg.path.data_dir) / "analyses"
    results_dir.mkdir(exist_ok=True)

    if not cfg.logging.use_wandb:
        _plot_and_save_histories(cfg, histories)

    # Get CNN statistics and save them
    cnn_stats = cnn_statistics(
        device,
        test_set,
        brain,
        cfg.logging.channel_analysis,
        cfg.logging.plot_sample_size,
    )

    # Save CNN statistics
    with open(results_dir / f"cnn_stats_epoch_{epoch}.json", "w") as f:
        json.dump(asdict(cnn_stats), f, cls=NumpyEncoder)

    if epoch == 0:
        _perform_initialization_analysis(cfg, brain, objective, train_set, cnn_stats)

    _analyze_layers(cfg, cnn_stats, epoch, copy_checkpoint)

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
    cnn_stats: CNNStatistics,
):
    summary = brain.scan()
    filepath = Path(cfg.path.run_dir) / "brain_summary.txt"
    filepath.write_text(summary)

    if cfg.logging.use_wandb:
        wandb.save(str(filepath), base_path=cfg.path.run_dir, policy="now")

    # TODO: This is a bit of a hack, we should refactor this to get the relevant information out of  cnn_stats
    rf_sizes_fig = plot_receptive_field_sizes(**asdict(cnn_stats))
    _process_figure(cfg, False, rf_sizes_fig, init_dir, "receptive_field_sizes", 0)

    graph_fig = plot_brain_and_optimizers(brain, objective)
    _process_figure(cfg, False, graph_fig, init_dir, "brain_graph", 0)

    transforms = transform_base_images(train_set, num_steps=5, num_images=2)
    # Save transform statistics
    transform_path = Path(cfg.path.run_dir) / "results" / "transforms.json"
    with open(transform_path, "w") as f:
        json.dump(asdict(transforms), f, cls=NumpyEncoder)

    transforms_fig = plot_transforms(**asdict(transforms))
    _process_figure(cfg, False, transforms_fig, init_dir, "transforms", 0)

    _analyze_input_layer(cfg, cnn_stats.layers["input"], cfg.logging.channel_analysis)


def _analyze_layers(
    cfg: DictConfig,
    cnn_stats: CNNStatistics,
    epoch: int,
    copy_checkpoint: bool,
):
    for layer_name, layer_data in cnn_stats.layers.items():
        if layer_name != "input":
            _analyze_regular_layer(
                cfg,
                layer_name,
                layer_data,
                epoch,
                copy_checkpoint,
                cfg.logging.channel_analysis,
            )


def _analyze_input_layer(
    cfg: DictConfig,
    layer_statistics: LayerStatistics,
    channel_analysis: bool,
):
    layer_rfs = layer_receptive_field_plots(layer_statistics.receptive_fields)
    _process_figure(cfg, False, layer_rfs, init_dir, "input_rfs", 0)

    if channel_analysis:
        layer_dict = asdict(layer_statistics)
        num_channels = int(layer_dict.pop("num_channels"))
        for channel in range(num_channels):
            channel_fig = plot_channel_statistics(
                **layer_dict, layer_name="input", channel=channel
            )
            _process_figure(
                cfg, False, channel_fig, init_dir, f"input_channel_{channel}", 0
            )


def _analyze_regular_layer(
    cfg: DictConfig,
    layer_name: str,
    layer_statistics: LayerStatistics,
    epoch: int,
    copy_checkpoint: bool,
    channel_analysis: bool,
):
    layer_rfs = layer_receptive_field_plots(layer_statistics.receptive_fields)
    _process_figure(
        cfg, copy_checkpoint, layer_rfs, "receptive_fields", f"{layer_name}", epoch
    )

    if channel_analysis:
        layer_dict = asdict(layer_statistics)
        num_channels = int(layer_dict.pop("num_channels"))
        for channel in range(num_channels):
            channel_fig = plot_channel_statistics(
                **layer_dict, layer_name=layer_name, channel=channel
            )

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
        norm_means, norm_stds = train_set.normalization_stats
        rec_dict = asdict(
            reconstruct_images(device, brain, decoder, train_set, test_set, 5)
        )
        recon_fig = plot_reconstructions(
            norm_means,
            norm_stds,
            *rec_dict["train"].values(),
            *rec_dict["test"].values(),
            num_samples=5,
        )
        _process_figure(
            cfg,
            copy_checkpoint,
            recon_fig,
            "reconstruction",
            f"{decoder}_reconstructions",
            epoch,
        )


### Helper Functions ###


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
    if cfg.logging.use_wandb:
        title = f"{_wandb_title(sub_dir)}/{_wandb_title(file_name)}"
        img = wandb.Image(fig)
        wandb.log({title: img}, commit=False)
    else:
        _save_figure(cfg, sub_dir, file_name, fig)
        if copy_checkpoint:
            _checkpoint_copy(cfg, sub_dir, file_name, epoch)
    plt.close(fig)
