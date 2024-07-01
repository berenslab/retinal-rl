import os
import sys
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from retinal_rl.classification.util import delete_results, initialize
from retinal_rl.models.brain import Brain
from runner.analyze import analyze
from runner.train import train


@hydra.main(config_path="config/base", config_name="config", version_base=None)
def program(cfg: DictConfig):
    if cfg.command.run_mode == "clean":
        delete_results(cfg.system.experiment_path, cfg.system.data_path)
        sys.exit(0)

    device = torch.device(cfg.system.device)
    brain = Brain(**cfg.brain).to(device)
    optimizer = torch.optim.Adam(brain.parameters(), lr=cfg.command.learning_rate)

    # Load CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    cache_path = os.path.join(hydra.utils.get_original_cwd(), "cache")
    train_set: Dataset[Tuple[Tensor, int]] = CIFAR10(
        root=cache_path, train=True, download=True, transform=transform
    )
    test_set: Dataset[Tuple[Tensor, int]] = CIFAR10(
        root=cache_path, train=False, download=True, transform=transform
    )

    if cfg.command.run_mode == "scan":
        brain.scan_circuits()
        sys.exit(0)

    brain, optimizer, histories, completed_epochs = initialize(
        cfg.system.data_path,
        cfg.system.checkpoint_path,
        cfg.system.plot_path,
        brain,
        optimizer,
    )

    if cfg.command.run_mode == "train":
        train(
            cfg,
            device,
            brain,
            optimizer,
            train_set,
            test_set,
            completed_epochs,
            histories,
        )
        sys.exit(0)

    if cfg.command.run_mode == "analyze":
        analyze(
            cfg,
            device,
            brain,
            histories,
            train_set,
            test_set,
        )
        sys.exit(0)

    raise ValueError("Invalid run_mode")


if __name__ == "__main__":
    program()
