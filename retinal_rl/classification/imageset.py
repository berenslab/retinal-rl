"""Provides a flexible Imageset class for image processing tasks, applicable to both classification and reinforcement learning scenarios.

It includes:
- Imageset: A flexible wrapper class for image datasets that applies transformations
            and handles dataset multiplication or on-the-fly transformations.
"""

import logging
import time
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
        # Use len() instead of iterating through entire dataset
        self.base_len = len(self.base_dataset)

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
        """Convert a PIL image to a PyTorch tensor"""
        tensor: Tensor = tf.to_tensor(img)
        return tensor

    def normalize_maybe(self, img: Tensor) -> Tensor:
        """Normalize the image tensor if needed"""
        if self.apply_normalization:
            mean, std = self.normalization_stats
            img = tf.normalize(img, mean, std)
        return img

    def epoch_len(self) -> int:
        """Get the length of the dataset for one epoch. For fixed transformations, this is the base length times the multiplier. For on-the-fly transformations, this is the length of the base dataset."""
        if self.fixed_transformation:
            return self.base_len * self.multiplier
        return self.base_len

    def __len__(self) -> int:
        return self.epoch_len()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        # Add global timing tracking
        if not hasattr(self, '_getitem_times'):
            self._getitem_times = []
            self._getitem_count = 0
        
        start_time = time.time()
        
        if self.fixed_transformation:
            result = self.transformed_dataset[idx]
        else:
            # For on-the-fly transformations
            img, label = self.base_dataset[idx]

            # Convert to Tensor
            img_tensor = self.to_tensor(img)

            # Apply source transformations
            source_tensor = self.source_transforms(img_tensor)
            noisy_tensor = self.normalize_maybe(
                self.noise_transforms(source_tensor.clone())
            )
            source_tensor = self.normalize_maybe(source_tensor)

            result = source_tensor, noisy_tensor, label
        
        # Track timing
        elapsed = time.time() - start_time
        self._getitem_times.append(elapsed)
        self._getitem_count += 1
        
        # Print summary periodically
        if self._getitem_count % 1000 == 0:
            avg_time = sum(self._getitem_times[-1000:]) / min(1000, len(self._getitem_times))
            print(f"  __getitem__ avg time (last 1000): {avg_time:.6f}s")
        
        return result


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
