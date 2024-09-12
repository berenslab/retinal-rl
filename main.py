"""Main entry point for the retinal RL project."""

import sys

import hydra
import torch
from omegaconf import DictConfig

from retinal_rl.classification.objective import ClassificationContext
from retinal_rl.models.brain import Brain
from retinal_rl.models.optimizer import BrainOptimizer
from runner.analyze import analyze
from runner.dataset import get_datasets
from runner.debug import check_parameter_overlap, compare_gradient_computation
from runner.initialize import initialize
from runner.sweep import launch_sweep
from runner.train import train
from runner.util import delete_results


# Hydra entry point
@hydra.main(config_path="config", config_name="config", version_base=None)
def _program(cfg: DictConfig):
    if cfg.command == "clean":
        delete_results(cfg)
        sys.exit(0)

    if cfg.command == "sweep":
        launch_sweep(cfg)
        sys.exit(0)

    device = torch.device(cfg.system.device)

    brain = Brain(**cfg.experiment.brain).to(device)
    brain_optimizer = BrainOptimizer[ClassificationContext](
        brain, dict(cfg.experiment.optimizer)
    )

    if cfg.command == "scan":
        brain.scan_circuits()
        # brain.visualize_connectome()
        sys.exit(0)

    train_set, test_set = get_datasets(cfg)

    brain, brain_optimizer, histories, completed_epochs = initialize(
        cfg,
        brain,
        brain_optimizer,
    )
    # Sanity checking
    check_parameter_overlap(brain_optimizer)

    # Debug mode operations
    if cfg.command == "debug":
        print("Running debug checks...")

        print("\nComparing gradient computation methods:")
        gradients_match, discrepancies = compare_gradient_computation(
            device, brain, brain_optimizer, train_set
        )

        if gradients_match:
            print("All gradients match within tolerance.")
        else:
            print("Discrepancies found in gradient computation:")
            for param, diff in discrepancies.items():
                if diff is None:
                    print(f"  {param}: Mismatch (one gradient is None)")
                else:
                    print(f"  {param}: {diff}")

        print("\nDebug checks completed.")
        sys.exit(0)

    if cfg.command == "train":
        train(
            cfg,
            device,
            brain,
            brain_optimizer,
            train_set,
            test_set,
            completed_epochs,
            histories,
        )
        sys.exit(0)

    if cfg.command == "analyze":
        analyze(cfg, device, brain, histories, train_set, test_set, completed_epochs)
        sys.exit(0)

    raise ValueError("Invalid run_mode")


if __name__ == "__main__":
    _program()
