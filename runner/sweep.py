from omegaconf import DictConfig, OmegaConf

import wandb


def launch_sweep(cfg: DictConfig):
    # Convert the relevant parts of the config to a dictionary
    sweep_config = OmegaConf.to_container(cfg.sweep, resolve=False)

    # Initialize wandb
    wandb.login()  # Ensure you're logged in to wandb

    # Launch the sweep
    wandb.sweep(sweep=sweep_config, project=sweep_config["project"])
