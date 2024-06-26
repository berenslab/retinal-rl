### Imports ###

import os
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

import hydra
from omegaconf import DictConfig

from retinal_rl.classification.training import single_run, cross_validate, save_results

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def train(cfg: DictConfig):

    # Load CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    cache_path = os.path.join(hydra.utils.get_original_cwd(), "cache")
    dataset = CIFAR10(root=cache_path, train=True, download=True, transform=transform)
    device = torch.device(cfg.system.device)


    if cfg.command.run_mode == "single_run":
        brain, history = single_run(device, cfg, cfg.command.num_epochs,
                                    cfg.command.recon_weight, dataset)
        save_results([brain], [history])
    elif cfg.run_mode == "cross_validate":
        brains, histories = cross_validate (device, cfg
                                            , cfg.command.num_folds,
                                           cfg.command.num_epochs, cfg.command.recon_weight, dataset)
        save_results(brains, histories)
    else:
        raise ValueError("Invalid run_mode")

if __name__ == "__main__":
    train()
