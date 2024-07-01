from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset

from retinal_rl.classification.training import evaluate_model, run_epoch
from retinal_rl.classification.util import save_checkpoint
from retinal_rl.models.brain import Brain
from runner.analyze import analyze, checkpoint_analyze


def print_histories(histories: dict[str, List[float]]) -> None:
    """Prints the training and test histories in a readable format."""
    print(f"Train Total Loss: {histories['train_total'][-1]:.4f}")
    print(f"Train Classification Loss: {histories['train_classification'][-1]:.4f}")
    print(f"Train Reconstruction Loss: {histories['train_reconstruction'][-1]:.4f}")
    print(f"Test Total Loss: {histories['test_total'][-1]:.4f}")
    print(f"Test Classification Loss: {histories['test_classification'][-1]:.4f}")
    print(f"Test Reconstruction Loss: {histories['test_reconstruction'][-1]:.4f}")


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
        train_loss, train_recon_loss, train_class_loss = evaluate_model(
            device,
            brain,
            cfg.command.recon_weight,
            recon_objective,
            class_objective,
            trainloader,
        )
        test_loss, test_recon_loss, test_class_loss = evaluate_model(
            device,
            brain,
            cfg.command.recon_weight,
            recon_objective,
            class_objective,
            testloader,
        )

        histories["train_total"].append(train_loss)
        histories["train_classification"].append(train_class_loss)
        histories["train_reconstruction"].append(train_recon_loss)
        histories["test_total"].append(test_loss)
        histories["test_classification"].append(test_class_loss)
        histories["test_reconstruction"].append(test_recon_loss)

    print(f"\nInitialization complete. Performance at Epoch {completed_epochs}:")
    print_histories(histories)

    for epoch in range(
        completed_epochs + 1, completed_epochs + cfg.command.num_epochs + 1
    ):
        brain, histories = run_epoch(
            device,
            brain,
            histories,
            cfg.command.recon_weight,
            optimizer,
            class_objective,
            recon_objective,
            trainloader,
            testloader,
        )

        print(f"\nEpoch {epoch} complete.")
        print_histories(histories)

        if epoch % cfg.command.checkpoint_step == 0:
            print("Saving checkpoint and plots.")

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
            checkpoint_analyze(
                checkpoint_plot_path,
                brain,
                train_set,
                test_set,
                device,
            )
            analyze(
                cfg,
                brain,
                histories,
                train_set,
                test_set,
                device,
            )
