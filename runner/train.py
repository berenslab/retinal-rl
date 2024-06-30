from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset

from retinal_rl.classification.training import run_epoch
from retinal_rl.classification.util import save_results
from retinal_rl.models.brain import Brain
from runner.analyze import analyze, checkpoint_analyze


def print_histories(histories: dict[str, List[float]], epoch: int) -> None:
    """Prints the training and test histories in a readable format."""
    print(f"Epoch {epoch}:")
    print(f"Train Total Loss: {histories['train_total'][-1]:.4f}")
    print(f"Train Classification Loss: {histories['train_classification'][-1]:.4f}")
    print(f"Train Reconstruction Loss: {histories['train_reconstruction'][-1]:.4f}")
    print(f"Test Total Loss: {histories['test_total'][-1]:.4f}")
    print(f"Test Classification Loss: {histories['test_classification'][-1]:.4f}")
    print(f"Test Reconstruction Loss: {histories['test_reconstruction'][-1]:.4f}")


def train(
    cfg: DictConfig,
    brain: Brain,
    train_set: Dataset[Tuple[Tensor, int]],
    test_set: Dataset[Tuple[Tensor, int]],
    device: torch.device,
    completed_epochs: int,
    histories: Dict[str, List[float]],
):
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = DataLoader(test_set, batch_size=64, shuffle=False)

    class_objective = nn.CrossEntropyLoss()
    recon_objective = nn.MSELoss()
    optimizer = optim.Adam(brain.parameters(), lr=0.001)

    for i in range(1, cfg.command.num_epochs + 1):
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

        print_histories(histories, i + completed_epochs)

        if i % cfg.command.checkpoint_step == 0:
            print(f"Completed epoch {i+completed_epochs}. Saving checkpoint and plots.")
            checkpoint_plot_path = (
                f"{cfg.system.checkpoint_plot_path}/checkpoint-epoch-{i+completed_epochs}"
            )

            save_results(
                cfg.system.data_path,
                cfg.system.checkpoint_path,
                cfg.system.max_checkpoints,
                brain,
                histories,
            )
            checkpoint_analyze(
                checkpoint_plot_path,
                brain,
                train_set,
                test_set,
                device,
            )
            analyze(
                cfg.system.plot_path,
                cfg.command.plot_inputs,
                brain,
                histories,
                train_set,
                test_set,
                device,
            )
