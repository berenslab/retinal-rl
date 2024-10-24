"""Provides a flexible Imageset class for image processing tasks, applicable to both classification and reinforcement learning scenarios.

It includes:
- Imageset: A flexible wrapper class for image datasets that applies transformations
            and handles dataset multiplication or on-the-fly transformations.
"""

import logging
from typing import List, Sequence, Tuple

import torchvision.transforms.functional as tf
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


class Imageset(Dataset[Tuple[Tensor, Tensor, int]]):
    """A flexible wrapper class for image datasets that applies transformations and handles dataset multiplication or on-the-fly transformations."""

    def __init__(
        self,
        base_dataset: Dataset[Tuple[Image.Image, int]],
        source_transforms: List[nn.Module] = [],
        noise_transforms: List[nn.Module] = [],
        apply_normalization: bool = True,
        normalization_mean: List[float] = [0.5, 0.5, 0.5],
        normalization_std: List[float] = [0.5, 0.5, 0.5],
        fixed_transformation: bool = False,
        multiplier: int = 1,
    ) -> None:
        """Initialize the Imageset.

        Args:
        ----
            base_dataset (Dataset): The base dataset to wrap.
            source_transforms (Optional[List[nn.Module]]): List of transformations to apply to the source image.
            noise_transforms (Optional[List[nn.Module]]): List of additional transformations to apply as noise.
            apply_normalization (bool): Whether to apply normalization after other transforms.
            normalization_stats (Tuple[List[float], List[float]]):
                Mean and std for normalization. Default is ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).
            fixed_transformation (bool): Whether to apply transformations once and store results (True)
                                         or apply them on-the-fly (False).
            multiplier (int): Number of times to multiply the dataset (only used when fixed_transformation is True).

        """
        self.base_dataset = base_dataset
        self.source_transforms = nn.Sequential(*source_transforms)
        self.noise_transforms = nn.Sequential(*noise_transforms)
        self.apply_normalization = apply_normalization
        self.normalization_stats = (normalization_mean, normalization_std)
        self.fixed_transformation = fixed_transformation
        self.multiplier = multiplier if fixed_transformation else 1
        self.base_len = 0
        for _ in self.base_dataset:
            self.base_len += 1

        if fixed_transformation:
            self.transformed_dataset = self._create_fixed_dataset()

    def _create_fixed_dataset(self) -> List[Tuple[Tensor, Tensor, int]]:
        transformed_data: List[Tuple[Tensor, Tensor, int]] = []
        for img, label in self.base_dataset:
            for _ in range(self.multiplier):
                source_img = self.source_transforms(img)
                noisy_img = self.noise_transforms(source_img)
                transformed_data.append(
                    (self.to_tensor(source_img), self.to_tensor(noisy_img), label)
                )
        return transformed_data

    def to_tensor(self, img: Image.Image) -> Tensor:
        """Convert a PIL image to a PyTorch tensor and apply normalization if needed."""
        tensor: Tensor = tf.to_tensor(img)
        if self.apply_normalization:
            mean, std = self.normalization_stats
            tensor = tf.normalize(tensor, mean, std)
        return tensor

    def epoch_len(self) -> int:
        """Get the length of the dataset for one epoch. For fixed transformations, this is the base length times the multiplier. For on-the-fly transformations, this is the length of the base dataset."""
        if self.fixed_transformation:
            return self.base_len * self.multiplier
        return self.base_len

    def __len__(self) -> int:
        return self.epoch_len()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        if self.fixed_transformation:
            return self.transformed_dataset[idx]

        # For on-the-fly transformations
        img, label = self.base_dataset[idx]

        # Apply source transformations
        source_img = self.source_transforms(img)
        noisy_img = self.noise_transforms(source_img)

        # Convert to tensor and normalize
        source_tensor = self.to_tensor(source_img)
        noisy_tensor = self.to_tensor(noisy_img)

        return source_tensor, noisy_tensor, label


class ImageSubset(Subset[Tuple[Tensor, Tensor, int]]):
    """A simple subset class that can be used to create a subset of any dataset."""

    def __init__(
        self, dataset: Dataset[Tuple[Tensor, Tensor, int]], indices: Sequence[int]
    ) -> None:
        """Initialize the ImageSubset.

        Args:
        ----
            dataset (Dataset[Tuple[Tensor, Tensor, int]]): The original dataset.
            indices (Sequence[int]): Sequence of indices to include in the subset.

        """
        super().__init__(dataset, indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        return super().__getitem__(idx)

    def __len__(self):
        return len(self.indices)
