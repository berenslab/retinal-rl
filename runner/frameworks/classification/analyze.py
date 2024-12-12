import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig

from retinal_rl.analysis import channel_analysis as channel_ana
from retinal_rl.analysis import receptive_fields
from retinal_rl.analysis.plot import (
    FigureLogger,
    plot_brain_and_optimizers,
    plot_receptive_field_sizes,
)
from retinal_rl.analysis.reconstructions import perform_reconstruction_analysis
from retinal_rl.analysis.transforms_analysis import (
    plot_transforms,
    transform_base_images,
)
from retinal_rl.classification.imageset import Imageset
from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import ContextT, Objective
from retinal_rl.util import FloatArray, NumpyEncoder

### Infrastructure ###

init_dir = "initialization_analysis"


@dataclass
class AnalysesCfg:
    run_dir: Path
    plot_dir: Path
    checkpoint_plot_dir: Path
    data_dir: Path
    use_wandb: bool
    channel_analysis: bool
    plot_sample_size: int

    def __post_init__(self):
        self.analyses_dir = Path(self.data_dir) / "analyses"

        # Ensure all dirs exist
        self.run_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
        self.checkpoint_plot_dir.mkdir(exist_ok=True)
        self.analyses_dir.mkdir(exist_ok=True)


### Analysis ###


def analyze(
    cfg: DictConfig,
    device: torch.device,
    brain: Brain,
    objective: Objective[ContextT],
    histories: dict[str, list[float]],
    train_set: Imageset,
    test_set: Imageset,
    epoch: int,
    copy_checkpoint: bool = False,
):
    ## DictConfig

    _cfg = AnalysesCfg(
        Path(cfg.path.run_dir),
        Path(cfg.path.plot_dir),
        Path(cfg.path.checkpoint_plot_dir),
        Path(cfg.path.data_dir),
        cfg.logging.use_wandb,
        cfg.logging.channel_analysis,
        cfg.logging.plot_sample_size,
    )
    log = FigureLogger(
        _cfg.use_wandb, _cfg.plot_dir, _cfg.checkpoint_plot_dir, _cfg.run_dir
    )

    ## Analysis
    log.plot_and_save_histories(histories)

    # # Save CNN statistics # TODO: how to do this now...
    # with open(_cfg.analyses_dir / f"cnn_stats_epoch_{epoch}.json", "w") as f:
    #     json.dump(asdict(cnn_stats), f, cls=NumpyEncoder)

    # perform different analyses
    input_shape, rf_result = receptive_fields.analyze(brain, device)
    receptive_fields.plot(
        log,
        rf_result,
        epoch,
        copy_checkpoint,
    )

    if _cfg.channel_analysis:
        spectral_result = channel_ana.spectral_analysis(
            device, test_set, brain, _cfg.plot_sample_size
        )
        histogram_result = channel_ana.histogram_analysis(
            device, test_set, brain, _cfg.plot_sample_size
        )
        # TODO: Do we really want to replot rfs here?
        channel_ana.plot(
            log,
            rf_result,
            spectral_result,
            histogram_result,
            epoch,
            copy_checkpoint,
        )
    else:
        spectral_result, histogram_result = None, None

    # plot results
    if epoch == 0:
        _default_initialization_plots(log, brain, objective, input_shape, rf_result)
        _extended_initialization_plots(
            log,
            _cfg.channel_analysis,
            _cfg.analyses_dir,
            train_set,
            rf_result,
            spectral_result,
            histogram_result,
        )

    perform_reconstruction_analysis(
        log,
        _cfg.analyses_dir,
        device,
        brain,
        objective,
        train_set,
        test_set,
        epoch,
        copy_checkpoint,
    )

    log.plot_and_save_histories(histories, save_always=True)


def _default_initialization_plots(
    log: FigureLogger,
    brain: Brain,
    objective: Objective[ContextT],
    input_shape: tuple[int, ...],
    rf_result: dict[str, FloatArray],
):
    log.save_summary(brain)

    # TODO: Move this somewhere accessible for RL
    # TODO: This is a bit of a hack, we should refactor this to get the relevant information out of  cnn_stats
    rf_sizes_fig = plot_receptive_field_sizes(input_shape, rf_result)
    log.log_figure(
        rf_sizes_fig,
        init_dir,
        "receptive_field_sizes",
        0,
        False,
    )

    graph_fig = plot_brain_and_optimizers(brain, objective)
    log.log_figure(
        graph_fig,
        init_dir,
        "brain_graph",
        0,
        False,
    )


def _extended_initialization_plots(
    log: FigureLogger,
    channel_analysis: bool,
    analyses_dir: Path,
    train_set: Imageset,
    rf_result: dict[str, FloatArray],
    spectral_result: Optional[dict[str, channel_ana.SpectralAnalysis]] = None,
    histogram_result: Optional[dict[str, channel_ana.HistogramAnalysis]] = None,
):
    transforms = transform_base_images(train_set, num_steps=5, num_images=2)
    # Save transform statistics
    transform_path = analyses_dir / "transforms.json"
    with open(transform_path, "w") as f:
        json.dump(asdict(transforms), f, cls=NumpyEncoder)

    transforms_fig = plot_transforms(**asdict(transforms))
    log.log_figure(
        transforms_fig,
        init_dir,
        "transforms",
        0,
        False,
    )

    if spectral_result and histogram_result:
        _analyze_input_layer(
            log,
            rf_result["input"],
            spectral_result["input"],
            histogram_result["input"],
            channel_analysis,
        )


def _analyze_input_layer(
    log: FigureLogger,
    rf_result: FloatArray,
    spectral_result: channel_ana.SpectralAnalysis,
    histogram_result: channel_ana.HistogramAnalysis,
    channel_analysis: bool,
):
    layer_rfs = receptive_fields.layer_receptive_field_plots(rf_result)
    log.log_figure(
        layer_rfs,
        init_dir,
        "input_rfs",
        0,
        False,
    )  # TODO: What's the purpose of it - it's just RGB I guess?
    if channel_analysis:
        for channel in range(rf_result.shape[0]):
            channel_fig = channel_ana.layer_channel_plots(
                rf_result,
                spectral_result,
                histogram_result,
                layer_name="input",
                channel=channel,
            )
            log.log_figure(
                channel_fig,
                init_dir,
                f"input_channel_{channel}",
                0,
                False,
            )
