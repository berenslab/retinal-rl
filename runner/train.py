import logging
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from retinal_rl.classification.training import process_dataset, run_epoch
from retinal_rl.models.brain import Brain
from runner.analyze import analyze
from runner.util import save_checkpoint

# Initialize the logger
log = logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    device: torch.device,
    brain: Brain,
    train_set: Dataset[Tuple[Tensor, int]],
    test_set: Dataset[Tuple[Tensor, int]],
    completed_epochs: int,
    history: Dict[str, List[float]],
):
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = DataLoader(test_set, batch_size=64, shuffle=False)

    class_objective = nn.CrossEntropyLoss()
    recon_objective = nn.MSELoss()
    wall_time = time.time()
    epoch_wall_time = 0

    if completed_epochs == 0:
        brain.train()
        train_losses = process_dataset(
            device, brain, class_objective, recon_objective, trainloader, is_training=True
        )
        brain.eval()
        test_losses = process_dataset(
            device, brain, class_objective, recon_objective, testloader, is_training=False
        )

        # Initialize the history
        for key in train_losses:
            history[f"train_{key}"] = [train_losses[key]]
        for key in test_losses:
            history[f"test_{key}"] = [test_losses[key]]

        analyze(
            cfg,
            device,
            brain,
            history,
            train_set,
            test_set,
            completed_epochs,
            True,
        )

        if cfg.logging.use_wandb:
            _log_statistics(completed_epochs, epoch_wall_time, history)

    log.info("Initialization complete.")

    for epoch in range(completed_epochs + 1, cfg.training.num_epochs + 1):
        brain, history = run_epoch(
            device,
            brain,
            history,
            class_objective,
            recon_objective,
            trainloader,
            testloader,
        )

        new_wall_time = time.time()
        epoch_wall_time = new_wall_time - wall_time
        wall_time = new_wall_time
        log.info(f"Epoch {epoch} complete. Epoch Wall Time: {epoch_wall_time:.2f}s.")

        if epoch % cfg.training.checkpoint_step == 0:
            log.info("Saving checkpoint and plots.")

            save_checkpoint(
                cfg.system.data_dir,
                cfg.system.checkpoint_dir,
                cfg.system.max_checkpoints,
                brain,
                history,
                epoch,
            )

            analyze(
                cfg,
                device,
                brain,
                history,
                train_set,
                test_set,
                epoch,
                True,
            )

        if cfg.logging.use_wandb:
            _log_statistics(epoch, epoch_wall_time, history)


def _log_statistics(
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
