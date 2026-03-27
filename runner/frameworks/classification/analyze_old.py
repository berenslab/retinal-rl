from dataclasses import asdict, dataclass
from pathlib import Path
import logging
import time

import numpy as np
import torch

from retinal_rl.analysis import channel_analysis as channel_ana
from retinal_rl.analysis import default as default_ana
from retinal_rl.analysis import fit_analysis
from retinal_rl.analysis import latent_visualisation as latent_ana
from retinal_rl.analysis import receptive_fields
from retinal_rl.analysis import reconstructions as recon_ana
from retinal_rl.analysis import transforms_analysis as transf_ana
from retinal_rl.analysis.dog_fit_analysis import dog_map_from_params, fit_dog_2d
from retinal_rl.analysis.gabor_fit_analysis import gabor_map_from_params, fit_gabor_2d
from retinal_rl.analysis.plot import FigureLogger
from retinal_rl.math_utils import FloatArray
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
    batch_size: int = 64
    fit_analysis: bool = False
    fit_blur_sigma: float = 0.5
    latent_analysis: bool = False
    latent_layer: str = "visual_cortex"
    channel_plot_epoch_step: int = 5

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
    logger = logging.getLogger(__name__)
    log = FigureLogger(
        cfg.use_wandb, cfg.plot_dir, cfg.checkpoint_plot_dir, cfg.run_dir
    )

    analysis_start_time = time.time()

    log.plot_and_save_histories(histories)

    # ============ Receptive Fields ============
    start_time = time.time()
    input_shape, rf_result = receptive_fields.analyze(brain, device)
    receptive_fields.plot(
        log,
        rf_result,
        epoch,
        copy_checkpoint,
    )
    log.save_dict(cfg.analyses_dir / f"receptive_fields_epoch_{epoch}.npz", rf_result)
    rf_time = time.time() - start_time
    logger.info(f"Epoch {epoch} - Receptive fields: {rf_time:.2f}s")

    # ============ Fit Analysis (DoG + Gabor) ============
    fit_time = 0.0
    if cfg.fit_analysis:
        start_time = time.time()
        run_fit_analysis(
            log, rf_result, cfg.analyses_dir, epoch, copy_checkpoint,
            blur_sigma=cfg.fit_blur_sigma,
        )
        fit_time = time.time() - start_time
        logger.info(f"Epoch {epoch} - Fit analysis (DoG + Gabor): {fit_time:.2f}s")

    # ============ Channel Analysis (Spectral + Histogram) ============
    channel_time = 0.0
    if cfg.channel_analysis:
        start_time = time.time()
        # Prepare dataset
        dataloader = channel_ana.prepare_dataset(test_set, cfg.plot_sample_size, cfg.batch_size)
        spectral_result = channel_ana.spectral_analysis(device, dataloader, brain)
        histogram_result = channel_ana.histogram_analysis(device, dataloader, brain)

        # Gate plotting based on epoch frequency
        should_plot = (epoch == 0) or (
            cfg.channel_plot_epoch_step > 0
            and epoch % cfg.channel_plot_epoch_step == 0
        )
        if should_plot:
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
        )  # TODO: Check if compressed save possible
        log.save_dict(
            cfg.analyses_dir / f"histogram_stats_epoch_{epoch}.npz", histogram_result
        )
        channel_time = time.time() - start_time
        logger.info(f"Epoch {epoch} - Channel analysis (spectral + histogram): {channel_time:.2f}s")
    else:
        spectral_result, histogram_result = None, None

    # ============ Latent Analysis (t-SNE) ============
    latent_time = 0.0
    if cfg.latent_analysis:
        start_time = time.time()
        tsne_results, labels = latent_ana.analyze(
            device,
            brain,
            test_set,
            layer_name=cfg.latent_layer,
            max_samples=cfg.plot_sample_size,
            batch_size=cfg.batch_size,
        )
        if tsne_results is not None:
            latent_ana.plot(log, tsne_results, labels, epoch, copy_checkpoint, cfg.latent_layer)
        latent_time = time.time() - start_time
        logger.info(f"Epoch {epoch} - Latent analysis (t-SNE): {latent_time:.2f}s")

    # ============ Reconstruction Analysis ============
    start_time = time.time()
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
    recon_time = time.time() - start_time
    logger.info(f"Epoch {epoch} - Reconstruction analysis: {recon_time:.2f}s")

    # ============ Summary ============
    total_analysis_time = time.time() - analysis_start_time
    logger.info(
        f"Epoch {epoch} - ANALYSIS SUMMARY (total {total_analysis_time:.2f}s):\n"
        f"  - Receptive fields:    {rf_time:6.2f}s ({100*rf_time/total_analysis_time:5.1f}%)\n"
        f"  - Fit analysis:        {fit_time:6.2f}s ({100*fit_time/total_analysis_time:5.1f}%)\n"
        f"  - Channel analysis:    {channel_time:6.2f}s ({100*channel_time/total_analysis_time:5.1f}%)\n"
        f"  - Latent analysis:     {latent_time:6.2f}s ({100*latent_time/total_analysis_time:5.1f}%)\n"
        f"  - Reconstruction:      {recon_time:6.2f}s ({100*recon_time/total_analysis_time:5.1f}%)"
    )


    if epoch == 0:
        default_ana.initialization_plots(log, brain, objective, input_shape, rf_result)
        # Skip expensive channel analysis at init (will run at epoch 0 via sparse schedule)
        # _extended_initialization_plots(
        #     log,
        #     cfg.channel_analysis,
        #     cfg.analyses_dir,
        #     input_shape,
        #     train_set,
        #     cfg.plot_sample_size,
        #     cfg.batch_size,
        #     device,
        # )

    log.plot_and_save_histories(histories, save_always=True)


def _extended_initialization_plots(
    log: FigureLogger,
    channel_analysis: bool,
    analyses_dir: Path,
    input_shape: tuple[int, ...],
    train_set: Imageset,
    max_sample_size: int,
    batch_size: int,
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

def run_fit_analysis(
    log: FigureLogger,
    rf_result: dict[str, FloatArray],
    analyses_dir: Path,
    epoch: int,
    copy_checkpoint: bool,
    blur_sigma: float = 0.5,
) -> None:
    """Run both DoG and Gabor fit analysis."""
    for key, display_name, fit_2d_fn, map_fn in [
        ("dog", "DoG", fit_dog_2d, dog_map_from_params),
        ("gabor", "Gabor", fit_gabor_2d, gabor_map_from_params),
    ]:
        results = fit_analysis.analyze_all_layers(rf_result, fit_2d_fn, map_fn, blur_sigma)
        npz = fit_analysis.prepare_npz_dict(results)
        log.save_dict(analyses_dir / f"{key}_fits_epoch_{epoch}.npz", npz)

        r2_history_path = analyses_dir / f"{key}_r2_history.npz"
        r2_history = fit_analysis.update_and_save_r2_history(r2_history_path, results, epoch)

        fit_analysis.plot(log, rf_result, results, epoch, copy_checkpoint, r2_history, display_name)


