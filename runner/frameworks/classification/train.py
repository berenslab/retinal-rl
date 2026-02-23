"""Training loop for the Brain."""

import logging
import time
from pathlib import Path
from typing import Dict, List

import torch
import wandb
from omegaconf import DictConfig
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from retinal_rl.classification.imageset import Imageset
from retinal_rl.classification.loss import ClassificationContext
from retinal_rl.classification.training import process_dataset, run_epoch
from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import Objective
from runner.frameworks.classification.analyze import AnalysesCfg, analyze
from runner.util import save_checkpoint

# Initialize the logger
logger = logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    device: torch.device,
    brain: Brain,
    objective: Objective[ClassificationContext],
    optimizer: Optimizer,
    train_set: Imageset,
    test_set: Imageset,
    initial_epoch: int,
    history: Dict[str, List[float]],
):
    """Train the Brain model using the specified optimizer.

    Args:
    ----
        cfg (DictConfig): The configuration for the experiment.
        device (torch.device): The device to run the computations on.
        brain (Brain): The Brain model to train and evaluate.
        objective (Objective): The optimizer for updating the model parameters.
        train_set (Imageset): The training dataset.
        test_set (Imageset): The test dataset.
        initial_epoch (int): The epoch to start training from.
        history (Dict[str, List[float]]): The training history.

    """

    use_wandb = cfg.logging.use_wandb

    data_dir = Path(cfg.path.data_dir)
    checkpoint_dir = Path(cfg.path.checkpoint_dir)

    max_checkpoints = cfg.logging.max_checkpoints
    checkpoint_step = cfg.logging.checkpoint_step

    num_epochs = cfg.optimizer.num_epochs
    num_workers = cfg.system.num_workers

    trainloader = DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=num_workers
    )
    testloader = DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=num_workers
    )

    wall_time = time.time()

    if initial_epoch == 0:
        brain.train()
        train_losses = process_dataset(
            device,
            brain,
            objective,
            optimizer,
            initial_epoch,
            trainloader,
            is_training=False,
        )
        brain.eval()
        test_losses = process_dataset(
            device,
            brain,
            objective,
            optimizer,
            initial_epoch,
            testloader,
            is_training=False,
        )

        # Initialize the history
        logger.info("Epoch 0 training performance:")
        for key, value in train_losses.items():
            logger.info(f"{key}: {value:.4f}")
            history[f"train_{key}"] = [value]
        for key, value in test_losses.items():
            history[f"test_{key}"] = [value]
        ana_cfg = AnalysesCfg(
            run_dir=Path(cfg.path.run_dir),
            plot_dir=Path(cfg.path.plot_dir),
            checkpoint_plot_dir=Path(cfg.path.checkpoint_plot_dir),
            data_dir=Path(cfg.path.data_dir),
            use_wandb=cfg.logging.use_wandb,
            channel_analysis=cfg.logging.channel_analysis,
            plot_sample_size=cfg.logging.plot_sample_size,
            fit_analysis=cfg.logging.get("fit_analysis", False),
            fit_blur_sigma=cfg.logging.get("fit_blur_sigma", 0.5),
        )
        analyze(
            ana_cfg,
            device,
            brain,
            objective,
            history,
            train_set,
            test_set,
            initial_epoch,
            True,
        )

        new_wall_time = time.time()
        epoch_wall_time = new_wall_time - wall_time
        wall_time = new_wall_time
        logger.info(f"Initialization complete. Wall Time: {epoch_wall_time:.2f}s.")

        if use_wandb:
            _wandb_log_statistics(initial_epoch, epoch_wall_time, history)

    else:
        logger.info(
            f"Reloading complete. Resuming training from epoch {initial_epoch}."
        )

    for epoch in range(initial_epoch + 1, num_epochs + 1):
        brain, history = run_epoch(
            device,
            brain,
            objective,
            optimizer,
            history,
            epoch,
            trainloader,
            testloader,
        )

        new_wall_time = time.time()
        epoch_wall_time = new_wall_time - wall_time
        wall_time = new_wall_time
        logger.info(f"Epoch {epoch} complete. Wall Time: {epoch_wall_time:.2f}s.")

        if epoch % checkpoint_step == 0:
            logger.info("Saving checkpoint.")

            save_checkpoint(
                data_dir,
                checkpoint_dir,
                max_checkpoints,
                brain,
                optimizer,
                history,
                epoch,
            )

        ana_cfg = AnalysesCfg(
            run_dir=Path(cfg.path.run_dir),
            plot_dir=Path(cfg.path.plot_dir),
            checkpoint_plot_dir=Path(cfg.path.checkpoint_plot_dir),
            data_dir=Path(cfg.path.data_dir),
            use_wandb=cfg.logging.use_wandb,
            channel_analysis=cfg.logging.channel_analysis,
            plot_sample_size=cfg.logging.plot_sample_size,
            fit_analysis=cfg.logging.get("fit_analysis", False),
            fit_blur_sigma=cfg.logging.get("fit_blur_sigma", 0.5),
        )

        analyze(
            ana_cfg,
            device,
            brain,
            objective,
            history,
            train_set,
            test_set,
            epoch,
            epoch % checkpoint_step == 0,
        )
        logger.info("Analysis complete.")

        if use_wandb:
            _wandb_log_statistics(epoch, epoch_wall_time, history)


def _wandb_log_statistics(
    epoch: int, epoch_wall_time: float, histories: Dict[str, List[float]]
) -> None:
    log_dict = {
        "Epoch": epoch,
        "Auxiliary/Epoch Wall Time": epoch_wall_time,
    }

    for key, values in histories.items():
        # Split the key into category (train/test) and metric name
        category, *metric_parts = key.split("_")
        metric_name = " ".join(word.capitalize() for word in metric_parts)

        # Create the full log key
        log_key = f"{category.capitalize()}/{metric_name}"

        # Add to log dictionary
        log_dict[log_key] = values[-1]

    wandb.log(log_dict, commit=True)
