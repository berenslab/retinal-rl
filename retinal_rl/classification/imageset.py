"""Provides a flexible ImageSet class for applying transformations to tensor datasets."""

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


class ImageSet(Dataset[Tuple[Tensor, Tensor, int]]):
    """Dataset wrapper that applies transformations to tensor-based datasets.

    For fixed transformations (multiplier > 1), transforms are applied once at initialization
    and the results are stored. For dynamic transformations (multiplier = 1), transforms
    are applied on-the-fly during __getitem__.
    """

    def __init__(
        self,
        base_dataset: Dataset[Tuple[Tensor, int]],
        source_transforms: List[nn.Module] = [],
        noise_transforms: List[nn.Module] = [],
        normalization_mean: Optional[list[float]] = None,
        normalization_std: Optional[list[float]] = None,
        fixed_transforms: bool = False,
        multiplier: int = 1,
    ) -> None:
        """Initialize the ImageSet.

        Args:
            base_dataset: Tensor-based dataset providing (image, label) pairs
            source_transforms: List of transformations to apply to source images
            noise_transforms: List of transformations for additional noise
            fixed_transforms: If True, pre-compute fixed transformations at initialization.
            multiplier: Ignored if fixed_transforms is False. Number of times to repeat each image.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.source_transforms = nn.Sequential(*source_transforms)
        self.noise_transforms = nn.Sequential(*noise_transforms)
        self.multiplier = multiplier
        self.fixed_transforms = fixed_transforms
        self.norm_mean = normalization_mean
        self.norm_std = normalization_std
        self.base_len = 0
        for _ in self.base_dataset:
            self.base_len += 1

        # For fixed transformations, pre-compute all results
        if self.fixed_transforms:
            self.fixed_data = self._generate_fixed_transforms()

    def _apply_transforms(self, image: Tensor, transforms: nn.Sequential) -> Tensor:
        """Apply a sequence of transforms to an image."""
        with torch.no_grad():
            current = image
            for transform in transforms:
                if hasattr(transform, "transform"):
                    # For custom transforms that need randomization
                    factor = transform.trans_range[0] + torch.rand(1).item() * (
                        transform.trans_range[1] - transform.trans_range[0]
                    )
                    current = transform.transform(current, factor)
                else:
                    # For standard transforms
                    current = transform(current)
            return current

    def _generate_fixed_transforms(self) -> List[Tuple[Tensor, Tensor, int]]:
        """Generate all fixed transformations."""
        with torch.no_grad():
            fixed_data: List[Tuple[Tensor, Tensor, int]] = []
            for idx in range(self.base_len):
                image, label = self.base_dataset[idx]

                for _ in range(self.multiplier):
                    # Apply source transforms
                    source_img = self._apply_transforms(image, self.source_transforms)

                    # Apply noise transforms to the transformed source image
                    noisy_img = self._apply_transforms(
                        source_img, self.noise_transforms
                    )

                    fixed_data.append((source_img, noisy_img, label))

            return fixed_data

    def _denormalize(self, tensor: Tensor) -> Tensor:
        """Remove normalization."""
        if self.norm_mean is not None and self.norm_std is not None:
            return tensor * self.norm_std + self.norm_mean
        return tensor

    def _normalize(self, tensor: Tensor) -> Tensor:
        """Apply normalization."""
        if self.norm_mean is not None and self.norm_std is not None:
            return (tensor - self.norm_mean) / self.norm_std
        return tensor

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        """Get a single item from the dataset."""
        if self.fixed_transforms:
            # Return pre-computed fixed transformations
            return self.fixed_data[idx]

        # Apply dynamic transformations
        image, label = self.base_dataset[idx]

        # Apply source transforms
        source_img = self._apply_transforms(image, self.source_transforms)

        # Apply noise transforms to the transformed source image
        noisy_img = self._apply_transforms(source_img, self.noise_transforms)

        source_img = self._normalize(source_img)
        noisy_img = self._normalize(noisy_img)

        return source_img, noisy_img, label

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.fixed_transforms:
            return len(self.fixed_data)
        return self.base_len
