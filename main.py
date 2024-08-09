import os
import sys
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from retinal_rl.classification.dataset import ScaleShiftTransform
from retinal_rl.framework_interface import TrainingFramework
from retinal_rl.models.brain import Brain
from retinal_rl.rl.sample_factory.sf_framework import SFFramework
from runner.analyze import analyze
from runner.initialize import initialize
from runner.sweep import launch_sweep
from runner.train import train
from runner.util import delete_results

# Preamble
OmegaConf.register_new_resolver("eval", eval)


# Hydra entry point
@hydra.main(config_path="config/base", config_name="config", version_base=None)
def program(cfg: DictConfig):
    if cfg.command.run_mode == "clean":
        delete_results(cfg.system.experiment_path, cfg.system.data_path)
        sys.exit(0)

    if cfg.command.run_mode == "sweep":
        launch_sweep(cfg)
        sys.exit(0)

    del cfg.sweep

    device = torch.device(cfg.system.device)
    brain = Brain(**cfg.brain).to(device)
    optimizer = torch.optim.Adam(brain.parameters(), lr=cfg.training.learning_rate)

    if cfg.command.run_mode == "scan":
        brain.scan_circuits()
        # brain.visualize_connectome()
        sys.exit(0)
    
    framework: TrainingFramework

    cache_path = os.path.join(hydra.utils.get_original_cwd(), "cache")
    if cfg.framework == "rl":
        framework = SFFramework()
    else:
        #TODO: Make ClassifierEngine

        # Load CIFAR-10 dataset TODO: This should be ClassifierEngine.initialize
        transform = transforms.Compose(
            [
                ScaleShiftTransform(cfg.dataset.visual_field, cfg.dataset.scale_range),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set: Dataset[Tuple[Tensor, int]] = CIFAR10(
            root=cache_path, train=True, download=True, transform=transform
        )
        test_set: Dataset[Tuple[Tensor, int]] = CIFAR10(
            root=cache_path, train=False, download=True, transform=transform
        )

        brain, optimizer, histories, completed_epochs = initialize(
            cfg,
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
            analyze(cfg, device, brain, histories, train_set, test_set, completed_epochs)
            sys.exit(0)

        raise ValueError("Invalid run_mode")

    
    brain, optimizer, histories, completed_epochs = framework.initialize(
        cfg,
        brain,
        optimizer,
        data_root=cache_path
    )

    if cfg.command.run_mode == "train":
        framework.train(
            cfg,
            device,
            brain,
            optimizer,
            None,
            None,
            completed_epochs,
            histories
        )
        sys.exit(0)

    if cfg.command.run_mode == "analyze":
        framework.analyze(cfg, device, brain, histories, train_set, test_set, completed_epochs)
        sys.exit(0)

    raise ValueError("Invalid run_mode")

if __name__ == "__main__":
    program()
