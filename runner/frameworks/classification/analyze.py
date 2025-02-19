from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from retinal_rl.analysis import channel_analysis as channel_ana
from retinal_rl.analysis import default as default_ana
from retinal_rl.analysis import receptive_fields
from retinal_rl.analysis import reconstructions as recon_ana
from retinal_rl.analysis import transforms_analysis as transf_ana
from retinal_rl.analysis.plot import FigureLogger
from retinal_rl.classification.imageset import Imageset
from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import ContextT, Objective

### Infrastructure ###


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
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_plot_dir.mkdir(parents=True, exist_ok=True)
        self.analyses_dir.mkdir(parents=True, exist_ok=True)


### Analysis ###


def analyze(
    cfg: AnalysesCfg,
    device: torch.device,
    brain: Brain,
    objective: Objective[ContextT],
    histories: dict[str, list[float]],
    train_set: Imageset,
    test_set: Imageset,
    epoch: int,
    copy_checkpoint: bool = False,
):
    log = FigureLogger(
        cfg.use_wandb, cfg.plot_dir, cfg.checkpoint_plot_dir, cfg.run_dir
    )

    log.plot_and_save_histories(histories)

    # perform different analyses, plot and log them
    input_shape, rf_result = receptive_fields.analyze(brain, device)
    receptive_fields.plot(
        log,
        rf_result,
        epoch,
        copy_checkpoint,
    )
    log.save_dict(cfg.analyses_dir / f"receptive_fields_epoch_{epoch}.npz", rf_result)

    if cfg.channel_analysis:
        # Prepare dataset
        dataloader = channel_ana.prepare_dataset(test_set, cfg.plot_sample_size)
        spectral_result = channel_ana.spectral_analysis(device, dataloader, brain)
        histogram_result = channel_ana.histogram_analysis(device, dataloader, brain)
        channel_ana.plot(
            log,
            rf_result,
            spectral_result,
            histogram_result,
            epoch,
            copy_checkpoint,
        )
        log.save_dict(
            cfg.analyses_dir / f"spectral_stats_epoch_{epoch}.npz", spectral_result
        ) # TODO: Check if compressed save possible
        log.save_dict(
            cfg.analyses_dir / f"histogram_stats_epoch_{epoch}.npz", spectral_result
        )
    else:
        spectral_result, histogram_result = None, None

    res, means, stds = recon_ana.analyze(device, brain, objective, train_set, test_set)
    recon_ana.plot(
        log,
        cfg.analyses_dir,
        res,
        means,
        stds,
        epoch,
        copy_checkpoint,
    )

    if epoch == 0:
        default_ana.initialization_plots(log, brain, objective, input_shape, rf_result)
        _extended_initialization_plots(
            log,
            cfg.channel_analysis,
            cfg.analyses_dir,
            input_shape,
            train_set,
            cfg.plot_sample_size,
            device,
        )

    log.plot_and_save_histories(histories, save_always=True)


def _extended_initialization_plots(
    log: FigureLogger,
    channel_analysis: bool,
    analyses_dir: Path,
    input_shape: tuple[int, ...],
    train_set: Imageset,
    max_sample_size: int,
    device: torch.device,
):
    transforms = transf_ana.analyze(train_set, num_steps=5, num_images=2)
    # Save transform statistics
    log.save_dict(analyses_dir / "transforms.npz", asdict(transforms))

    transforms_fig = transf_ana.plot(**asdict(transforms))
    log.log_figure(
        transforms_fig,
        default_ana.INIT_DIR,
        "transforms",
        0,
        False,
    )

    if channel_analysis:
        # Input 'rfs' is just the colors
        rf_result = np.eye(input_shape[0])[:, :, np.newaxis, np.newaxis]
        dataloader = channel_ana.prepare_dataset(train_set, max_sample_size)
        spectral_result, histogram_result = channel_ana.analyze_input(
            device, dataloader
        )
        channel_ana.input_plot(
            log,
            rf_result,
            spectral_result,
            histogram_result,
            default_ana.INIT_DIR,
        )
