"""Handles the configuration and initialization of datasets for experiments."""

from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import hydra
import torch
import torchvision.transforms.functional as tf
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm

from retinal_rl.classification.imageset import ImageSet


class CachedDataset(Dataset[Tuple[Tensor, int]]):
    """Abstract base class for cached tensor datasets."""

    def __init__(
        self,
        name: str,
        cache_dir: Path,
        train: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the cached dataset.

        Args:
            name: Name of the dataset
            cache_dir: Root directory for saving cache files
            train: If True, creates dataset from training set
            device: Device to store tensors on
        """
        super().__init__()
        self.name = name
        self.cache_dir = cache_dir
        self.train = train
        self.device = device or torch.device("cpu")
        self.tensor_cache_dir = self.cache_dir / "tensor_cache"
        self.tensor_cache_dir.mkdir(exist_ok=True)

        # Initialize these properly in subclasses after loading
        self.data: Tensor
        self.targets: list[int]

    def _load_or_create_cache(self) -> Tuple[Tensor, list[int]]:
        """Load tensors from cache or create if they don't exist."""
        cache_path = self._get_cache_path()

        if cache_path.exists():
            print(f"Loading {self.name} from tensor cache")
            cached = torch.load(str(cache_path))
            return cached["data"].to(self.device), cached["targets"]

        print(f"Creating tensor cache for {self.name}")
        data, targets = self._create_tensor_cache()

        # Save to cache
        cache_dict = {"data": data.cpu(), "targets": targets}
        torch.save(cache_dict, str(cache_path))
        return data.to(self.device), targets

    @abstractmethod
    def _create_tensor_cache(self) -> Tuple[Tensor, list[int]]:
        """Convert dataset to tensors. Must be implemented by subclasses."""
        pass

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Get a single item from the dataset."""
        return self.data[idx], self.targets[idx]

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.data)

    def _get_cache_path(self) -> Path:
        """Get the cache file path."""
        split = "train" if self.train else "test"
        return self.tensor_cache_dir / f"{self.name}_{split}.pt"


class CachedCIFAR10(CachedDataset):
    """CIFAR10 dataset that uses a tensor-based cache for faster loading."""

    name = "cifar10"

    def __init__(
        self,
        cache_dir: Path,
        train: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            name="cifar10", cache_dir=cache_dir, train=train, device=device
        )
        self.data, self.targets = self._load_or_create_cache()

    def _create_tensor_cache(self) -> Tuple[Tensor, list[int]]:
        """Convert CIFAR10 dataset to tensors."""
        dataset = datasets.CIFAR10(
            root=self.cache_dir,
            train=self.train,
            download=True,
        )

        # Pre-allocate tensor for images
        tensors = torch.empty((len(dataset), 3, 32, 32), dtype=torch.float32)
        labels: list[int] = []

        # Convert to tensors
        for idx in tqdm(range(len(dataset)), desc="Converting to tensors"):
            img, label = dataset[idx]
            tensors[idx] = tf.to_tensor(img)  # type: ignore[arg-type]
            labels.append(label)

        return tensors, labels


def get_datasets(cfg: DictConfig) -> Tuple[ImageSet, ImageSet]:
    """Get the train and test datasets, using tensor caching when possible."""
    cache_dir = Path(hydra.utils.get_original_cwd()) / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Get device from config
    device = torch.device(cfg.system.device)

    if cfg.dataset.name.upper() != "CIFAR10":
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    # Create datasets
    train_base = CachedCIFAR10(cache_dir=cache_dir, train=True, device=device)
    test_base = CachedCIFAR10(cache_dir=cache_dir, train=False, device=device)

    # Wrap with transforms
    train_set = ImageSet(
        base_dataset=train_base,
        source_transforms=cfg.dataset.source_transforms,
        noise_transforms=cfg.dataset.noise_transforms,
    )
    test_set = ImageSet(
        base_dataset=test_base,
        source_transforms=cfg.dataset.source_transforms,
        noise_transforms=cfg.dataset.noise_transforms,
    )

    return train_set, test_set
