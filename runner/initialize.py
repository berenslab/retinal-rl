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
logger = logging.getLogger(__name__)


def initialize(
    cfg: DictConfig,
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    wandb_sweep_id = os.getenv("WANDB_SWEEP_ID", "local")
    logger.info(f"Run Name: {cfg.run_name}")
    logger.info(f"(WANDB) Sweep ID: {wandb_sweep_id}")

    # If continuing from a previous run, load the model and history
    if os.path.exists(cfg.system.data_dir):
        return initialize_reload(cfg, brain, optimizer)
    # else, initialize a new model and history
    return initialize_create(cfg, brain, optimizer)


def initialize_reload(
    cfg: DictConfig, brain: Brain, optimizer: Optimizer
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    logger.info(
        f"Experiment dir {cfg.system.run_dir} exists. Loading existing model and history."
    )
    checkpoint_file = os.path.join(cfg.system.data_dir, "current_checkpoint.pt")

    # check if files don't exist
    if not os.path.exists(checkpoint_file):
        logger.error(f"File not found: {checkpoint_file}")
        raise FileNotFoundError("Checkpoint file does not exist.")

    # Load the state dict into the brain model
    checkpoint = torch.load(checkpoint_file)
    brain.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epochs = checkpoint["completed_epochs"]
    history = checkpoint["training_history"]

    if cfg.logging.use_wandb:
        wandb.init(
            project="retinal-rl",
            group=cfg.experiment,
            job_type=cfg.brain.name,
            name=cfg.run_name,
            id=cfg.run_name,
            resume="must",
        )
        wandb.mark_preempting()

    return brain, optimizer, history, completed_epochs


def initialize_create(
    cfg: DictConfig,
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    completed_epochs = 0
    logger.info(
        f"Experiment path {cfg.system.run_dir} does not exist. Initializing {cfg.run_name}."
    )
    history: Dict[str, List[float]] = {
        "train_total": [],
        "train_classification": [],
        "train_fraction_correct": [],
        "train_reconstruction": [],
        "test_total": [],
        "test_classification": [],
        "test_fraction_correct": [],
        "test_reconstruction": [],
    }
    # create the directories
    os.makedirs(cfg.system.data_dir)
    os.makedirs(cfg.system.checkpoint_dir)
    os.makedirs(cfg.system.plot_dir)

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
            name=cfg.run_name,
            id=cfg.run_name,
        )
        wandb.mark_preempting()
        wandb.define_metric("Epoch")
        wandb.define_metric("Train/*", step_metric="Epoch")
        wandb.define_metric("Test/*", step_metric="Epoch")

    return brain, optimizer, history, completed_epochs
