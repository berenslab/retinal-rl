import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset

from retinal_rl.classification.training import evaluate_model, run_epoch
from retinal_rl.models.brain import Brain
from runner.util import save_checkpoint
from runner.analyze import analyze

import wandb


# Initialize the logger
log = logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    device: torch.device,
    brain: Brain,
    optimizer: optim.Optimizer,
    train_set: Dataset[Tuple[Tensor, int]],
    test_set: Dataset[Tuple[Tensor, int]],
    completed_epochs: int,
    histories: Dict[str, List[float]],
):
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = DataLoader(test_set, batch_size=64, shuffle=False)

    class_objective = nn.CrossEntropyLoss()
    recon_objective = nn.MSELoss()

    if completed_epochs == 0:
        train_loss, train_class_loss, train_frac_correct, train_recon_loss = (
            evaluate_model(
                device,
                brain,
                cfg.training.recon_weight,
                recon_objective,
                class_objective,
                trainloader,
            )
        )
        test_loss, test_class_loss, test_frac_correct, test_recon_loss = evaluate_model(
            device,
            brain,
            cfg.training.recon_weight,
            recon_objective,
            class_objective,
            testloader,
        )

        histories["train_total"].append(train_loss)
        histories["train_classification"].append(train_class_loss)
        histories["train_fraction_correct"].append(train_frac_correct)
        histories["train_reconstruction"].append(train_recon_loss)
        histories["test_total"].append(test_loss)
        histories["test_classification"].append(test_class_loss)
        histories["test_fraction_correct"].append(test_frac_correct)
        histories["test_reconstruction"].append(test_recon_loss)

        checkpoint_plot_path = (
            f"{cfg.system.checkpoint_plot_path}/checkpoint-epoch-{completed_epochs}"
        )
        analyze(
            cfg,
            device,
            brain,
            histories,
            train_set,
            test_set,
            completed_epochs,
            check_path=checkpoint_plot_path,
        )

        if cfg.logging.use_wandb:
            _log_histories(cfg, histories, completed_epochs)

    log.info(f"Initialization complete.")

    for epoch in range(
        completed_epochs + 1, completed_epochs + cfg.training.num_epochs + 1
    ):
        brain, histories = run_epoch(
            device,
            brain,
            histories,
            cfg.training.recon_weight,
            optimizer,
            class_objective,
            recon_objective,
            trainloader,
            testloader,
        )

        log.info(f"Epoch {epoch} complete.")

        if epoch % cfg.training.checkpoint_step == 0:
            log.info("Saving checkpoint and plots.")

            save_checkpoint(
                cfg.system.data_path,
                cfg.system.checkpoint_path,
                cfg.system.max_checkpoints,
                brain,
                optimizer,
                histories,
                epoch,
            )

            checkpoint_plot_path = (
                f"{cfg.system.checkpoint_plot_path}/checkpoint-epoch-{epoch}"
            )
            analyze(
                cfg,
                device,
                brain,
                histories,
                train_set,
                test_set,
                epoch,
                check_path=checkpoint_plot_path,
            )

        if cfg.logging.use_wandb:
            _log_histories(cfg, histories, epoch)


def _log_histories(
    cfg: DictConfig, histories: dict[str, List[float]], epoch: int
) -> None:
    """Logs the training and test histories in a readable format."""
    # Flatten the hierarchical dictionary structure
    flat_dict = {
        "Train/Total Loss": histories["train_total"][-1],
        "Train/Classification Loss": histories["train_classification"][-1],
        "Train/Fraction Correct": histories["train_fraction_correct"][-1],
        "Train/Reconstruction Loss": histories["train_reconstruction"][-1],
        "Test/Total Loss": histories["test_total"][-1],
        "Test/Classification Loss": histories["test_classification"][-1],
        "Test/Fraction Correct": histories["test_fraction_correct"][-1],
        "Test/Reconstruction Loss": histories["test_reconstruction"][-1],
    }

    if cfg.logging.use_wandb:
        # Log to wandb if enabled
        wandb.log(flat_dict, step=epoch, commit=True)

    # Log to local logging in a pretty JSON format
    logging.debug(f"Training and Test Histories:\n{flat_dict}")
