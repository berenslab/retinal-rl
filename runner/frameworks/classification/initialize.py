"""Initialization functions."""
### Imports ###

import logging
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import omegaconf
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.optim.optimizer import Optimizer

import wandb
from retinal_rl.models.brain import Brain
from runner.util import save_checkpoint

### Infrastructure ###


# Initialize the logger
logger = logging.getLogger(__name__)


@dataclass
class InitConfig:
    """Configuration for initialization."""

    # Paths
    data_dir: Path
    checkpoint_dir: Path
    plot_dir: Path
    wandb_dir: Path

    # WandB settings
    use_wandb: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_preempt: bool

    # Run settings
    run_name: str
    max_checkpoints: int

    device: torch.device

    @classmethod
    def from_dict_config(cls, cfg: DictConfig) -> "InitConfig":
        """Create InitConfig from a DictConfig."""
        return cls(
            data_dir=Path(cfg.path.data_dir),
            checkpoint_dir=Path(cfg.path.checkpoint_dir),
            plot_dir=Path(cfg.path.plot_dir),
            wandb_dir=Path(cfg.path.wandb_dir),
            use_wandb=cfg.logging.use_wandb,
            wandb_project=cfg.logging.wandb_project,
            wandb_entity=None
            if cfg.logging.wandb_entity == "default"
            else cfg.logging.wandb_entity,
            wandb_preempt=cfg.logging.wandb_preempt,
            run_name=cfg.run_name,
            max_checkpoints=cfg.logging.max_checkpoints,
            device=cfg.system.device,
        )


### Initialization ###


def initialize(
    dict_cfg: DictConfig,
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    """Initialize the Brain, Optimizers, and training histories. Checks whether the experiment directory exists and loads the model and history if it does. Otherwise, initializes a new model and history."""

    cfg = InitConfig.from_dict_config(dict_cfg)
    wandb_sweep_id = getenv("WANDB_SWEEP_ID", "local")
    logger.info(f"Run Name: {cfg.run_name}")
    logger.info(f"(WANDB) Sweep ID: {wandb_sweep_id}")

    # If continuing from a previous run, load the model and history
    if cfg.data_dir.exists():
        return _initialize_reload(cfg, brain, optimizer)
    # else, initialize a new model and history
    logger.info(
        f"Experiment data path {cfg.data_dir} does not exist. Initializing {cfg.run_name}."
    )

    cfg_backup = omegaconf.OmegaConf.to_container(
        dict_cfg, resolve=True, throw_on_missing=True
    )
    cfg_backup = cast(Dict[str, Any], cfg_backup)

    return _initialize_create(cfg, cfg_backup, brain, optimizer)


def _initialize_create(
    cfg: InitConfig,
    cfg_backup: dict[Any, Any],
    brain: Brain,
    optimizer: Optimizer,
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    epoch = 0
    # initialize the training histories
    histories: Dict[str, List[float]] = {}

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.use_wandb:
        cfg.plot_dir.mkdir(parents=True, exist_ok=True)
    else:
        cfg.wandb_dir.mkdir(parents=True, exist_ok=True)
        # convert DictConfig to dict
        entity = cfg.wandb_entity
        if entity == "default":
            entity = None
        wandb.init(
            project=cfg.wandb_project,
            entity=entity,
            group=HydraConfig.get().runtime.choices.experiment,
            job_type=HydraConfig.get().runtime.choices.brain,
            config=cfg_backup,
            name=cfg.run_name,
            id=cfg.run_name,
            dir=cfg.wandb_dir,
        )

        if cfg.wandb_preempt:
            wandb.mark_preempting()

        wandb.define_metric("Epoch")
        wandb.define_metric("Train/*", step_metric="Epoch")
        wandb.define_metric("Test/*", step_metric="Epoch")

    save_checkpoint(
        cfg.data_dir,
        cfg.checkpoint_dir,
        cfg.max_checkpoints,
        brain,
        optimizer,
        histories,
        epoch,
    )

    return brain, optimizer, histories, epoch


def _initialize_reload(
    cfg: InitConfig, brain: Brain, optimizer: Optimizer
) -> Tuple[Brain, Optimizer, Dict[str, List[float]], int]:
    logger.info(
        f"Experiment data dir {cfg.data_dir} exists. Loading existing model and history."
    )
    checkpoint_file = cfg.data_dir / "current_checkpoint.pt"

    # check if files don't exist
    if not checkpoint_file.exists():
        logger.error(f"File not found: {checkpoint_file}")
        raise FileNotFoundError("Checkpoint file does not exist.")

    # Load the state dict into the brain model
    checkpoint = torch.load(checkpoint_file, map_location=cfg.device)
    brain.load_state_dict(checkpoint["brain_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epochs = checkpoint["completed_epochs"]
    history = checkpoint["histories"]
    entity = cfg.wandb_entity
    if entity == "default":
        entity = None

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=entity,
            group=HydraConfig.get().runtime.choices.experiment,
            job_type=HydraConfig.get().runtime.choices.brain,
            name=cfg.run_name,
            id=cfg.run_name,
            resume="must",
            dir=cfg.wandb_dir,
        )
        if cfg.wandb_preempt:
            wandb.mark_preempting()

    return brain, optimizer, history, completed_epochs
