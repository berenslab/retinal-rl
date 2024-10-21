"""Utility functions for launching wandb sweeps."""

from typing import Any, Dict, cast

from omegaconf import DictConfig, OmegaConf

import wandb


def launch_sweep(cfg: DictConfig):
    """Launch a wandb sweep using the provided configuration."""
    # Convert the relevant parts of the config to a dictionary
    sweep_config = OmegaConf.to_container(cfg.sweep, resolve=True)
    sweep_config = cast(Dict[str, Any], sweep_config)

    # Initialize wandb
    wandb.login()  # Ensure you're logged in to wandb

    # Launch the sweep
    wandb.sweep(sweep=sweep_config, project=sweep_config["project"])
