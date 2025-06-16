"""Main entry point for the retinal RL project."""

import os
import sys
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from runner.frameworks.classification.classification_framework import (
    ClassificationFramework,
)
from runner.frameworks.framework_interface import TrainingFramework
from runner.frameworks.rl.sf_framework import SFFramework
from runner.sweep import launch_sweep
from runner.util import create_brain, delete_results, load_brain_weights

# Load the eval resolver for OmegaConf
OmegaConf.register_new_resolver("eval", eval)


def get_framework(cfg: DictConfig, cache_path: str) -> TrainingFramework:
    framework_classes = {
        "rl": lambda: SFFramework(cfg, data_root=cache_path),
        "classification": lambda: ClassificationFramework(cfg),
    }

    try:
        return framework_classes[cfg.framework]()
    except KeyError:
        raise NotImplementedError(
            f"Framework '{cfg.framework}' is not implemented. Available frameworks: {list(framework_classes.keys())}"
        )


# Hydra entry point
@hydra.main(config_path="config/base", config_name="config", version_base=None)
def _program(cfg: DictConfig):
    # TODO: Instead of doing checks of the config here, we should implement
    # sth like the configstore which ensures config parameters are present
    print(cfg)

    if cfg.command == "clean":
        delete_results(cfg)
        sys.exit(0)

    if cfg.command == "sweep":
        launch_sweep(cfg)
        sys.exit(0)

    if cfg.command == "analyze":
        cfg.logging.use_wandb = False

    device = torch.device(cfg.system.device)

    brain = create_brain(cfg.brain).to(device)

    optimizer = instantiate(cfg.optimizer.optimizer, brain.parameters())
    if hasattr(cfg.optimizer, "objective"):
        objective = instantiate(cfg.optimizer.objective, brain=brain)
    else:
        warnings.warn("No objective specified, is that wanted?")

    if cfg.command == "scan":
        print(brain.scan())
        sys.exit(0)

    framework: TrainingFramework

    cache_path = os.path.join(hydra.utils.get_original_cwd(), "cache")

    framework = get_framework(cfg, cache_path)

    brain, optimizer = framework.initialize(brain, optimizer)

    # Load brain weights if specified - TODO: same for optimizer? framework specific loading?
    if hasattr(cfg, "init_weights_path"):
        load_brain_weights(brain, cfg.init_weights_path)

    if cfg.command == "train":
        framework.train(device, brain, optimizer, objective)
        sys.exit(0)

    if cfg.command == "analyze":
        framework.analyze(device, brain, objective)
        sys.exit(0)

    raise ValueError("Invalid run_mode")


if __name__ == "__main__":
    _program()
