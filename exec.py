### Imports ###

import os
import sys
from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from retinal_rl.classification.plot import (
    plot_input_distributions,
    plot_training_histories,
)
from retinal_rl.classification.training import run_epoch
from retinal_rl.classification.util import (
    delete_results,
    initialize,
    save_results,
)
from retinal_rl.models.analysis import get_reconstructions, gradient_receptive_fields
from retinal_rl.models.brain import Brain
from retinal_rl.models.plot import plot_reconstructions, receptive_field_plots


@hydra.main(config_path="config", config_name="config", version_base=None)
def program(cfg: DictConfig):
    if cfg.command.run_mode == "clean":
        delete_results(cfg.system.experiment_path, cfg.system.data_path)
        sys.exit(0)

    brain = Brain(**cfg.brain)

    if cfg.command.run_mode == "scan":
        brain.scan_circuits()
        sys.exit(0)

    brain, histories = initialize(
        cfg.system.data_path, cfg.system.checkpoint_path, cfg.system.plot_path, brain
    )
    completed_epochs = len(histories["train_total"])

    # Load CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    cache_path = os.path.join(hydra.utils.get_original_cwd(), "cache")
    dataset: Dataset[Tuple[Tensor, int]] = CIFAR10(
        root=cache_path, train=True, download=True, transform=transform
    )

    device = torch.device(cfg.system.device)

    if cfg.command.run_mode == "train":
        train_size = int(cfg.command.split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_set: Dataset[Tuple[Tensor, int]]
        validation_set: Dataset[Tuple[Tensor, int]]
        train_set, validation_set = random_split(dataset, [train_size, val_size])

        trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
        validationloader = DataLoader(validation_set, batch_size=64, shuffle=False)

        class_objective = nn.CrossEntropyLoss()
        recon_objective = nn.MSELoss()
        optimizer = optim.Adam(brain.parameters(), lr=0.001)

        for i in range(1, cfg.command.num_epochs + 1):
            brain, histories = run_epoch(
                device,
                brain,
                histories,
                cfg.command.recon_weight,
                optimizer,
                class_objective,
                recon_objective,
                trainloader,
                validationloader,
            )
            save_results(
                cfg.system.data_path,
                cfg.system.checkpoint_path,
                cfg.system.max_checkpoints,
                brain,
                histories,
            )
            if i % cfg.command.checkpoint_step == 0:
                print(f"Completed epoch {i+completed_epochs}. Saving checkpoint.")

        sys.exit(0)

    if cfg.command.run_mode == "analyze":
        hist_fig = plot_training_histories(histories)
        hist_path = os.path.join(cfg.system.plot_path, "histories")
        hist_fig.savefig(hist_path)
        plt.close()

        if cfg.command.plot_inputs:
            rgb_fig = plot_input_distributions(dataset)
            rgb_path = os.path.join(cfg.system.plot_path, "input-distributions")
            rgb_fig.savefig(rgb_path)
            plt.close()

        rf_dict = gradient_receptive_fields(device, brain.circuits["encoder"])
        for lyr, rfs in rf_dict.items():
            rf_fig = receptive_field_plots(rfs)
            rf_path = os.path.join(cfg.system.plot_path, f"{lyr}-layer-receptive-fields")
            rf_fig.savefig(rf_path)
            plt.close()

        imgs, recons = get_reconstructions(device, brain, dataset, 5)
        recon_fig = plot_reconstructions(imgs, recons, 5)
        recon_path = os.path.join(cfg.system.plot_path, "reconstructions")
        recon_fig.savefig(recon_path)

        sys.exit(0)

    raise ValueError("Invalid run_mode")


if __name__ == "__main__":
    program()
