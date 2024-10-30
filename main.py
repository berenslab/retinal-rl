"""Main entry point for the retinal RL project."""

import os
import sys
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from retinal_rl.framework_interface import TrainingFramework
from retinal_rl.rl.sample_factory.sf_framework import SFFramework
from runner.classification.classification_framework import ClassificationFramework
from runner.sweep import launch_sweep
from runner.util import create_brain, delete_results

# Load the eval resolver for OmegaConf
OmegaConf.register_new_resolver("eval", eval)


# Hydra entry point
@hydra.main(config_path="config/base", config_name="config", version_base=None)
def _program(cfg: DictConfig):
    # TODO: Instead of doing checks of the config here, we should implement
    # sth like the configstore which ensures config parameters are present

    if cfg.command == "clean":
        delete_results(cfg)
        sys.exit(0)

    if cfg.command == "sweep":
        launch_sweep(cfg)
        sys.exit(0)

    device = torch.device(cfg.system.device)

    brain = create_brain(cfg.brain).to(device)

    optimizer = instantiate(cfg.optimizer.optimizer, brain.parameters())
    if hasattr(cfg.optimizer, "objective"):
        objective = instantiate(cfg.optimizer.objective, brain=brain)
        # TODO: RL framework currently can't use objective
    else:
        warnings.warn("No objective specified, is that wanted?")

    if cfg.command == "scan":
        print(brain.scan())
        sys.exit(0)

    framework: TrainingFramework

    cache_path = os.path.join(hydra.utils.get_original_cwd(), "cache")
    if cfg.framework == "rl":
        framework = SFFramework(cfg, data_root=cache_path)
    elif cfg.framework == "classification":
        framework = ClassificationFramework(cfg)
    else:
        raise NotImplementedError(
            "only 'rl' or 'classification' framework implemented currently"
        )

    brain, optimizer = framework.initialize(brain, optimizer)

    if cfg.command == "train":
        framework.train(device, brain, optimizer, objective)
        sys.exit(0)

    if cfg.command == "analyze":
        framework.analyze(device, brain, objective)
        sys.exit(0)

    raise ValueError("Invalid run_mode")


if __name__ == "__main__":
    _program()
