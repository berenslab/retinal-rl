"""Initialization functions."""
### Imports ###

import logging
import os
from typing import Any, Dict, List, Tuple, cast

import omegaconf
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.optim.optimizer import Optimizer

from retinal_rl.models.brain import Brain
from runner.util import save_checkpoint

# Initialize the logger
logger = logging.getLogger(__name__)


def initialize(
    cfg: DictConfig,
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    """Initialize the Brain, Optimizers, and training histories. Checks whether the experiment directory exists and loads the model and history if it does. Otherwise, initializes a new model and history."""
    wandb_sweep_id = os.getenv("WANDB_SWEEP_ID", "local")
    logger.info(f"Run Name: {cfg.run_name}")
    logger.info(f"(WANDB) Sweep ID: {wandb_sweep_id}")

    # If continuing from a previous run, load the model and history
    if os.path.exists(cfg.path.data_dir):
        return _initialize_reload(cfg, brain, optimizer)
    # else, initialize a new model and history
    return _initialize_create(cfg, brain, optimizer)


def _initialize_create(
    cfg: DictConfig,
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    epoch = 0
    logger.info(
        f"Experiment path {cfg.path.run_dir} does not exist. Initializing {cfg.run_name}."
    )

    # initialize the training histories
    histories: Dict[str, List[float]] = {}

    # create the directories
    os.makedirs(cfg.path.data_dir)
    os.makedirs(cfg.path.checkpoint_dir)
    if not cfg.logging.use_wandb:
        os.makedirs(cfg.path.plot_dir)

    else:
        os.makedirs(cfg.path.wandb_dir)
        # convert DictConfig to dict
        dict_conf = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        dict_conf = cast(Dict[str, Any], dict_conf)
        entity = cfg.logging.wandb_entity
        if entity == "default":
            entity = None
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=entity,
            group=HydraConfig.get().runtime.choices.experiment,
            job_type=HydraConfig.get().runtime.choices.brain,
            config=dict_conf,
            name=cfg.run_name,
            id=cfg.run_name,
            dir=cfg.path.wandb_dir,
        )

        if cfg.logging.wandb_preempt:
            wandb.mark_preempting()

        wandb.define_metric("Epoch")
        wandb.define_metric("Train/*", step_metric="Epoch")
        wandb.define_metric("Test/*", step_metric="Epoch")

    save_checkpoint(
        cfg.path.data_dir,
        cfg.path.checkpoint_dir,
        cfg.logging.max_checkpoints,
        brain,
        optimizer,
        histories,
        epoch,
    )

    return brain, optimizer, histories, epoch


def _initialize_reload(
    cfg: DictConfig, brain: Brain, optimizer: Optimizer
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    logger.info(
        f"Experiment dir {cfg.path.run_dir} exists. Loading existing model and history."
    )
    checkpoint_file = os.path.join(cfg.path.data_dir, "current_checkpoint.pt")

    # check if files don't exist
    if not os.path.exists(checkpoint_file):
        logger.error(f"File not found: {checkpoint_file}")
        raise FileNotFoundError("Checkpoint file does not exist.")

    # Load the state dict into the brain model
    checkpoint = torch.load(checkpoint_file)
    brain.load_state_dict(checkpoint["brain_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epochs = checkpoint["completed_epochs"]
    history = checkpoint["histories"]
    entity = cfg.logging.wandb_entity
    if entity == "default":
        entity = None

    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=entity,
            group=HydraConfig.get().runtime.choices.experiment,
            job_type=HydraConfig.get().runtime.choices.brain,
            name=cfg.run_name,
            id=cfg.run_name,
            resume="must",
            dir=cfg.path.wandb_dir,
        )
        if cfg.logging.wandb_preempt:
            wandb.mark_preempting()

    return brain, optimizer, history, completed_epochs
