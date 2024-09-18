"""Provides custom dataset classes for image processing tasks, applicable to both classification and reinforcement learning scenarios.

It includes:
- Imageset: A flexible dataset class for image data that can apply different transformations
            to source and input images.
- ImageSubset: A subset of Imageset that maintains all transformation capabilities.
"""

import logging
from typing import List, Sized, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


class Imageset(Dataset[Tuple[Tensor, Tensor, int]], Sized):
    """A flexible dataset class for image data that can apply different transformations to source and input images, with built-in normalization and tensor conversion."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        source_transforms: List[Module] = [],
        noise_transforms: List[Module] = [],
        apply_normalization: bool = True,
        normalization_stats: Tuple[List[float], List[float]] = (
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ) -> None:
        """Initialize the Imageset.

        Args:
            image_paths (List[str]): List of paths to the images.
            labels (List[int]): List of labels corresponding to the images.
            source_transforms (List[Module]): Transformations to apply to the source image.
            Used to as the target for denoising autoencoders.
            noise_transforms (List[Module]): Additional transformations to apply to the input.
            apply_normalization (bool): Whether to apply normalization after other transforms.
            normalization_stats (Tuple[Tuple[float, float, float], Tuple[float, float, float]]):
                Mean and std for normalization. Default is ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).

        """
        self.image_paths = image_paths
        self.labels = labels
        self.source_transforms: nn.Sequential = nn.Sequential(*source_transforms)
        self.input_transforms: nn.Sequential = nn.Sequential(*noise_transforms)
        self.apply_normalization = apply_normalization
        self.normalization_stats = normalization_stats

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the original image
        img = Image.open(img_path).convert("RGB")

        # Apply source transformations if any
        source_img = self.source_transforms(img) if self.source_transforms else img

        # Apply input transformations if any
        input_img = (
            self.input_transforms(source_img) if self.input_transforms else source_img
        )

        # Convert to tensor
        source_tensor = tf.to_tensor(source_img)
        input_tensor = tf.to_tensor(input_img)

        # Apply normalization if enabled
        if self.apply_normalization:
            mean, std = self.normalization_stats
            source_tensor = tf.normalize(source_tensor, mean, std)
            input_tensor = tf.normalize(input_tensor, mean, std)

        return input_tensor, source_tensor, label


class ImageSubset(Subset[Tuple[Tensor, Tensor, int]], Imageset):
    """A subset of Imageset that maintains all transformation capabilities."""

    def __init__(self, dataset: Imageset, indices: List[int]) -> None:
        """Initialize the ImageSubset.

        Args:
            dataset (Imageset): The original dataset.
            indices (List[int]): List of indices to include in the subset.

        """
        super().__init__(dataset, indices)
        self.source_transforms = dataset.source_transforms
        self.input_transforms = dataset.input_transforms
        self.apply_normalization = dataset.apply_normalization
        self.normalization_stats = dataset.normalization_stats

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        return super().__getitem__(idx)


# Utility function
def create_image_subsets(
    dataset: Imageset, split_ratio: float = 0.8
) -> Tuple[ImageSubset, ImageSubset]:
    """Create train and test subsets from an Imageset.

    Args:
        dataset (Imageset): The original dataset to split.
        split_ratio (float): Ratio of data to use for training. Default is 0.8 (80% train, 20% test).

    Returns:
        Tuple[ImageSubset, ImageSubset]: Train and test subsets.

    """
    total_size = len(dataset)
    train_size = int(total_size * split_ratio)

    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_subset = ImageSubset(dataset, train_indices)
    test_subset = ImageSubset(dataset, test_indices)

    return train_subset, test_subset
