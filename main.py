"""Main entry point for the retinal RL project."""

import os
import sys
import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from runner.frameworks.classification.classification_framework import (
    ClassificationFramework,
)
from runner.frameworks.framework_interface import TrainingFramework
from runner.frameworks.rl.sf_framework import SFFramework
from runner.paths import BASE_CONFIG_DIR, BASE_CONFIG_NAME, load_base_config
from runner.resume import (
    ResumeRequest,
    effective_config,
    log_resume_to_wandb,
    resume_request,
)
from runner.sweep import launch_sweep
from runner.util import create_brain, delete_results

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


def _program(cfg: DictConfig, resume: ResumeRequest | None):
    # When resuming a run, use its stored config (+ new overrides) instead of
    # the freshly composed one; for fresh runs store the config snapshot.
    cfg, resume_entry = effective_config(cfg, resume)

    # TODO: Instead of doing checks of the config here, we should implement
    # sth like the configstore which ensures config parameters are present
    print(OmegaConf.to_yaml(cfg, resolve=True))

    OmegaConf.save(
        OmegaConf.create(cfg),
        Path(cfg.path.run_dir) / cfg.path.full_config,
        resolve=True,
    )

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

    # Also loads init_weights_path (only for fresh runs, not when resuming)
    brain, optimizer = framework.initialize(brain, optimizer)
    if resume_entry is not None and cfg.logging.use_wandb:
        log_resume_to_wandb(resume_entry)

    if cfg.command == "train":
        framework.train(device, brain, optimizer, objective)
        sys.exit(0)

    if cfg.command == "analyze":
        framework.analyze(device, brain, objective)
        sys.exit(0)

    raise ValueError("Invalid run_mode")


def main() -> None:
    # `python main.py run_name=<existing run> [overrides]` continues that run
    resume = resume_request(sys.argv, load_base_config(), Path.cwd())
    if resume is not None:
        print(f"Resuming run '{resume.run_name}' from {resume.files.run_dir}")
        sys.argv = resume.hydra_argv

    @hydra.main(
        config_path=str(BASE_CONFIG_DIR),
        config_name=BASE_CONFIG_NAME,
        version_base=None,
    )
    def hydra_entry(cfg: DictConfig) -> None:
        _program(cfg, resume)

    hydra_entry()


if __name__ == "__main__":
    main()
