"""Handles the configuration and initialization of datasets for experiments."""

import os

import hydra
from omegaconf import DictConfig
from torchvision import datasets

from retinal_rl.classification.imageset import Imageset


def get_datasets(cfg: DictConfig) -> tuple[Imageset, Imageset]:
    """Get the train and test datasets based on the configuration."""
    cache_dir = os.path.join(hydra.utils.get_original_cwd(), "cache")
    return _get_datasets(cache_dir, cfg.dataset.name, cfg.dataset.imageset)

def _get_datasets(cache_dir: str, dataset_name: str, imageset: DictConfig) -> tuple[Imageset, Imageset]:
    """Get the train and test datasets based on the configuration."""
    os.makedirs(cache_dir, exist_ok=True)

    # Load the base datasets
    if dataset_name.upper() == "CIFAR10":
        train_base = datasets.CIFAR10(root=cache_dir, train=True, download=True)
        test_base = datasets.CIFAR10(root=cache_dir, train=False, download=True)
    elif dataset_name.upper() == "MNIST":
        train_base = datasets.MNIST(root=cache_dir, train=True, download=True)
        test_base = datasets.MNIST(root=cache_dir, train=False, download=True)
    elif dataset_name.upper() == "SVHN":
        train_base = datasets.SVHN(root=cache_dir, split="train", download=True)
        test_base = datasets.SVHN(root=cache_dir, split="test", download=True)
    elif dataset_name.upper() == "RL_STREAM": # TODO: Reconsider if this is the approach to go for
        train_base = datasets.ImageFolder(root=cache_dir)
        test_base = datasets.ImageFolder(root=cache_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Instantiate the Imagesets using Hydra
    train_set = hydra.utils.instantiate(imageset, base_dataset=train_base)
    test_set = hydra.utils.instantiate(imageset, base_dataset=test_base)

    return train_set, test_set
