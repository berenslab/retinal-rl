### Imports ###

import logging
import os
from typing import Dict, List, Tuple

import omegaconf
import torch
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer

from retinal_rl.models.brain import Brain
from runner.util import get_wandb_sweep_id

# Initialize the logger
log = logging.getLogger(__name__)


def initialize(
    cfg: DictConfig,
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    completed_epochs = 0

    wandb_sweep_id = get_wandb_sweep_id()
    log.info(f"Run ID: {cfg.run_id}")
    log.info(f"(WANDB) Sweep ID: {wandb_sweep_id}")

    # If continuing from a previous run, load the model and history
    if os.path.exists(cfg.system.data_path):
        log.info("Data path exists. Loading existing model and history.")
        if cfg.logging.use_wandb:
            wandb.init(
                project="retinal-rl",
                group=cfg.experiment,
                job_type=cfg.brain.name,
                id=cfg.run_id,
            )

        brain, optimizer, history, completed_epochs = load_checkpoint(
            cfg.system.data_path, brain, optimizer
        )

    # else, initialize a new model and history
    else:
        log.info(
            f"Data path {cfg.system.data_path} does not exist. Initializing {cfg.run_id}."
        )
        history = initialize_histories()
        # create the directories
        os.makedirs(cfg.system.data_path)
        os.makedirs(cfg.system.checkpoint_path)
        os.makedirs(cfg.system.plot_path)

        if cfg.logging.use_wandb:
            # convert DictConfig to dict
            dict_conf = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            wandb.init(
                project="retinal-rl",
                group=cfg.experiment,
                job_type=cfg.brain.name,
                config=dict_conf,
                id=cfg.run_id,
                resume=True,
            )
            wandb.define_metric("Epoch")
            wandb.define_metric("Train/*", step_metric="Epoch")
            wandb.define_metric("Test/*", step_metric="Epoch")

    return brain, optimizer, history, completed_epochs


def load_checkpoint(
    data_path: str,
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    # Load histories from the JSON file
    checkpoint_file = os.path.join(data_path, "current_checkpoint.pt")

    # check if files don't exist
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"File not found: {checkpoint_file}")

    # Load the state dict into the brain model
    checkpoint = torch.load(checkpoint_file)
    brain.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epochs = checkpoint["completed_epochs"]
    history = checkpoint["training_history"]

    return brain, optimizer, history, completed_epochs


def initialize_histories() -> Dict[str, List[float]]:
    """Initialize an empty training history."""
    return {
        "train_total": [],
        "train_classification": [],
        "train_fraction_correct": [],
        "train_reconstruction": [],
        "test_total": [],
        "test_classification": [],
        "test_fraction_correct": [],
        "test_reconstruction": [],
    }
