"""Handles the configuration and initialization of datasets for experiments."""

import os
from typing import Tuple

import hydra
from omegaconf import DictConfig
from torchvision import datasets
from torchvision.transforms.v2 import RGB

from retinal_rl.classification.imageset import Imageset


def get_datasets(cfg: DictConfig) -> Tuple[Imageset, Imageset]:
    """Get the train and test datasets based on the configuration."""
    cache_dir = os.path.join(hydra.utils.get_original_cwd(), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load the base datasets
    if cfg.dataset.name.upper() == "CIFAR10":
        train_base = datasets.CIFAR10(root=cache_dir, train=True, download=True)
        test_base = datasets.CIFAR10(root=cache_dir, train=False, download=True)
    elif cfg.dataset.name.upper() == "MNIST":
        train_base = datasets.MNIST(root=cache_dir, train=True, download=True, transform=RGB())
        test_base = datasets.MNIST(root=cache_dir, train=False, download=True, transform=RGB())
    elif cfg.dataset.name.upper() == "SVHN":
        train_base = datasets.SVHN(root=cache_dir, split="train", download=True)
        test_base = datasets.SVHN(root=cache_dir, split="test", download=True)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.name}")

    # Instantiate the Imagesets using Hydra
    train_set = hydra.utils.instantiate(cfg.dataset.imageset, base_dataset=train_base)
    test_set = hydra.utils.instantiate(cfg.dataset.imageset, base_dataset=test_base)

    return train_set, test_set
