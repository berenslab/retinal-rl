import sys

import hydra
import torch
from omegaconf import DictConfig

from retinal_rl.models.brain import Brain
from runner.analyze import analyze
from runner.dataset import get_datasets
from runner.initialize import initialize
from runner.sweep import launch_sweep
from runner.train import train
from runner.util import delete_results


# Hydra entry point
@hydra.main(config_path="config/base", config_name="config", version_base=None)
def program(cfg: DictConfig):
    if cfg.command.run_mode == "clean":
        delete_results(cfg)
        sys.exit(0)

    if cfg.command.run_mode == "sweep":
        launch_sweep(cfg)
        sys.exit(0)

    device = torch.device(cfg.system.device)
    brain = Brain(**cfg.brain).to(device)
    optimizer = torch.optim.Adam(brain.parameters(), lr=cfg.training.learning_rate)

    if cfg.command.run_mode == "scan":
        brain.scan_circuits()
        # brain.visualize_connectome()
        sys.exit(0)

    train_set, test_set = get_datasets(cfg)

    brain, optimizer, histories, completed_epochs = initialize(
        cfg,
        brain,
        optimizer,
    )

    if cfg.command.run_mode == "train":
        train(
            cfg,
            device,
            brain,
            optimizer,
            train_set,
            test_set,
            completed_epochs,
            histories,
        )
        sys.exit(0)

    if cfg.command.run_mode == "analyze":
        analyze(cfg, device, brain, histories, train_set, test_set, completed_epochs)
        sys.exit(0)

    raise ValueError("Invalid run_mode")


if __name__ == "__main__":
    program()
