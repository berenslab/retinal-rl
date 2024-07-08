import os
from typing import Callable, Tuple

import hydra
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets, transforms

from retinal_rl.classification.dataset import ScaleShiftTransform

Imageset = Dataset[Tuple[Tensor, int]]


def get_datasets(
    cfg: DictConfig,
) -> Tuple[Imageset, Imageset]:
    transform = transforms.Compose(
        [
            ScaleShiftTransform(
                cfg.dataset.visual_field,
                cfg.dataset.image_rescale_range,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set, test_set = multiply_dataset(
        load_dataset_factory(cfg.dataset.name, transform),
        cfg.dataset.sample_size_multiplier,
    )

    return train_set, test_set


def multiply_dataset(
    dataset_factory: Callable[[], Tuple[Imageset, Imageset]], multiplier: int
) -> Tuple[Imageset, Imageset]:
    train_sets, test_sets = zip(*(dataset_factory() for _ in range(multiplier)))
    return ConcatDataset(train_sets), ConcatDataset(test_sets)


def download_dataset(name: str, cache_path: str):
    if name == "CIFAR10":
        datasets.CIFAR10(root=cache_path, download=True)
    elif name == "MNIST":
        datasets.MNIST(root=cache_path, download=True)
    elif name == "FASHIONMNIST":
        datasets.FashionMNIST(root=cache_path, download=True)
    elif name == "SVHN":
        datasets.SVHN(root=cache_path, download=True)


def load_dataset_factory(
    name: str, transform: transforms.Compose
) -> Callable[[], Tuple[Imageset, Imageset]]:
    cache_path = os.path.join(hydra.utils.get_original_cwd(), "cache")
    os.makedirs(cache_path, exist_ok=True)
    name = name.upper()
    download_dataset(name, cache_path)

    def dataset_factory():
        if name == "CIFAR10":
            train_set = datasets.CIFAR10(
                root=cache_path, train=True, download=False, transform=transform
            )
            test_set = datasets.CIFAR10(
                root=cache_path, train=False, download=False, transform=transform
            )
        elif name == "MNIST":
            train_set = datasets.MNIST(
                root=cache_path, train=True, download=False, transform=transform
            )
            test_set = datasets.MNIST(
                root=cache_path, train=False, download=False, transform=transform
            )
        elif name == "FASHIONMNIST":
            train_set = datasets.FashionMNIST(
                root=cache_path, train=True, download=False, transform=transform
            )
            test_set = datasets.FashionMNIST(
                root=cache_path, train=False, download=False, transform=transform
            )
        elif name == "SVHN":
            train_set = datasets.SVHN(root=cache_path, download=False)
            train_set = datasets.SVHN(
                root=cache_path, split="train", download=False, transform=transform
            )
            test_set = datasets.SVHN(
                root=cache_path, split="test", download=False, transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {name}")
        return train_set, test_set

    return dataset_factory
