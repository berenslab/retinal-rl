### Imports ###

import logging
import os
from typing import Dict, List, Tuple

import omegaconf
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer

import wandb
from retinal_rl.models.brain import Brain

# Initialize the logger
log = logging.getLogger(__name__)


def initialize(
    cfg: DictConfig,
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    completed_epochs = 0

    if os.path.exists(cfg.system.data_path):
        log.info("Data path exists. Loading existing model and history.")
        brain, optimizer, history, completed_epochs = load_checkpoint(
            cfg.system.data_path, brain, optimizer
        )
        if cfg.logging.use_wandb:
            wandb.init(project="retinal-rl")
    else:
        if cfg.logging.use_wandb:
            # convert DictConfig to dict
            dict_conf = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            wandb.init(project="retinal-rl", config=dict_conf)
        log.info("Data path does not exist. Initializing new model and history.")
        history = initialize_histories()
        # create the directories
        os.makedirs(cfg.system.data_path)
        os.makedirs(cfg.system.checkpoint_path)
        os.makedirs(cfg.system.plot_path)

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
