"""Main entry point for the retinal RL project."""

import os
import sys
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from retinal_rl.classification.loss import ClassificationContext
from retinal_rl.framework_interface import TrainingFramework
from retinal_rl.models.brain import Brain
from retinal_rl.models.goal import Goal
from retinal_rl.rl.sample_factory.sf_framework import SFFramework
from runner.analyze import analyze
from runner.dataset import get_datasets
from runner.initialize import initialize
from runner.sweep import launch_sweep
from runner.train import train
from runner.util import delete_results

OmegaConf.register_new_resolver("eval", eval)


# Hydra entry point
@hydra.main(config_path="config/base", config_name="config", version_base=None)
def _program(cfg: DictConfig):
    if cfg.command == "clean":
        delete_results(cfg)
        sys.exit(0)

    if cfg.command == "sweep":
        launch_sweep(cfg)
        sys.exit(0)

    device = torch.device(cfg.system.device)

    brain = Brain(**cfg.brain).to(device)
    if hasattr(cfg, "optimizer"):
        goal = Goal[ClassificationContext](brain, dict(cfg.optimizer.goal))
        optimizer = instantiate(cfg.optimizer.optimizer, parameters=brain.parameters())
    else:
        warnings.warn("No optimizer config specified, is that wanted?")

    if cfg.command == "scan":
        brain.scan_circuits()
        sys.exit(0)

    framework: TrainingFramework

    cache_path = os.path.join(hydra.utils.get_original_cwd(), "cache")
    if cfg.framework == "rl":
        framework = SFFramework(cfg, data_root=cache_path)
    else:
        # TODO: Make ClassifierEngine
        train_set, test_set = get_datasets(cfg)

        brain, optimizer, histories, completed_epochs = initialize(
            cfg,
            brain,
            optimizer,
        )
        if cfg.command == "train":
            train(
                cfg,
                device,
                brain,
                goal,
                optimizer,
                train_set,
                test_set,
                completed_epochs,
                histories,
            )
            sys.exit(0)

        if cfg.command == "analyze":
            analyze(
                cfg,
                device,
                brain,
                goal,
                histories,
                train_set,
                test_set,
                completed_epochs,
            )
            sys.exit(0)

        raise ValueError("Invalid run_mode")

    if cfg.command == "train":
        framework.train()
        sys.exit(0)

    if cfg.command == "analyze":
        framework.analyze(cfg, device, brain, histories, None, None, completed_epochs)
        sys.exit(0)

    raise ValueError("Invalid run_mode")


if __name__ == "__main__":
    _program()
