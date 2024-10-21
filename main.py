"""Main entry point for the retinal RL project."""

import os
import sys
import warnings
from typing import Dict, List, cast

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from retinal_rl.framework_interface import TrainingFramework
from retinal_rl.models.brain import Brain
from retinal_rl.rl.sample_factory.sf_framework import SFFramework
from runner.analyze import analyze
from runner.dataset import get_datasets
from runner.initialize import initialize
from runner.sweep import launch_sweep
from runner.train import train
from runner.util import assemble_neural_circuits, delete_results

# Load the eval resolver for OmegaConf
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

    sensors = OmegaConf.to_container(cfg.brain.sensors, resolve=True)
    sensors = cast(Dict[str, List[int]], sensors)

    connections = OmegaConf.to_container(cfg.brain.connections, resolve=True)
    connections = cast(List[List[str]], connections)

    connectome, circuits = assemble_neural_circuits(
        cfg.brain.circuits, sensors, connections
    )

    brain = Brain(circuits, sensors, connectome).to(device)

    if hasattr(cfg, "optimizer"):
        optimizer = instantiate(cfg.optimizer.optimizer, brain.parameters())
        objective = instantiate(cfg.optimizer.objective, brain=brain)
    else:
        warnings.warn("No optimizer config specified, is that wanted?")

    if cfg.command == "scan":
        brain.scan()
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
                objective,
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
                objective,
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
