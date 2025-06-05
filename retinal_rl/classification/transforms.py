"""Provides custom transformation classes for image processing tasks, applicable to both classification and reinforcement learning scenarios.

It includes various image transformations:
- ScaleShiftTransform
- ShotNoiseTransform
- ContrastTransform
- IlluminationTransform
- BlurTransform
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from torch import nn


class ContinuousTransform(nn.Module, ABC):
    """Base class for continuous image transformations."""

    def __init__(self, trans_range: Tuple[float, float]) -> None:
        """Initialize the ContinuousTransform."""
        super().__init__()
        self.trans_range: Tuple[float, float] = trans_range

    @property
    def name(self) -> str:
        """Return a pretty name of the transformation."""
        name = self.__class__.__name__
        # Remove the "Transform" suffix
        name = name.replace("Transform", "")
        # decamelcase
        return name.replace("([a-z])([A-Z])", r"\1 \2").lower()

    @abstractmethod
    def transform(self, img: torch.Tensor, trans_factor: float) -> torch.Tensor:
        """Apply the transformation to the input image.

        Args:
        ----
            img (torch.Tensor): The input image tensor to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            torch.Tensor: The transformed image tensor.

        """
        raise NotImplementedError

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Randomly apply the transformation to the input image.

        Args:
        ----
            img (torch.Tensor): The input image tensor to transform.

        Returns:
        -------
            torch.Tensor: The transformed image tensor.

        """
        trans_factor = np.random.uniform(self.trans_range[0], self.trans_range[1])
        return self.transform(img, trans_factor)


class IlluminationTransform(ContinuousTransform):
    """Apply random illumination (brightness) adjustment to the input image."""

    def __init__(self, brightness_range: Tuple[float, float]) -> None:
        """Initialize the IlluminationTransform.

        Args:
        ----
            brightness_range (Tuple[float, float]): Range of brightness adjustment factors. For an identity transform, set the range to (1, 1).

        """
        super().__init__(brightness_range)

    def transform(self, img: torch.Tensor, trans_factor: float) -> torch.Tensor:
        """Apply random illumination (brightness) adjustment to the input image.

        Args:
        ----
            img (torch.Tensor): The input image tensor to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            torch.Tensor: The transformed image tensor with adjusted illumination.

        """
        return TF.adjust_contrast(img, contrast_factor=trans_factor)


class BlurTransform(ContinuousTransform):
    """Apply random Gaussian blur to the input image."""

    def __init__(self, blur_range: Tuple[float, float], kernel_size: list[int] = 3) -> None:
        """Initialize the BlurTransform.

        Args:
        ----
            blur_range (Tuple[float, float]): Range of blur radii. For an identity transform, set the range to (0, 0).

        """
        super().__init__(blur_range)
        self.kernel_size = kernel_size

    def transform(self, img: torch.Tensor, trans_factor: float) -> torch.Tensor:
        """Apply random Gaussian blur to the input image.

        Args:
        ----
            img (torch.Tensor): The input image tensor to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            torch.Tensor: The transformed image tensor with applied blur.

        """
        blur_transform = T.GaussianBlur(kernel_size=int(trans_factor), sigma=(0.1, 2.0))
        TF.gaussian_blur(img, kernel_size=5)
        return blur_transform(img)


class ScaleShiftTransform(ContinuousTransform):
    """Apply random scale and shift transformations to the input image."""

    def __init__(
        self,
        vision_width: int,
        vision_height: int,
        image_rescale_range: Tuple[float, float],
    ) -> None:
        """Initialize the ScaleShiftTransform.

        Args:
        ----
            vision_width (int): The width of the visual field.
            vision_height (int): The height of the visual field.
            image_rescale_range (Tuple[float, float]): Range of image rescaling factors. For an identity transform, set the range to (1, 1).

        
        TODO: readd the following parameters (lost in merge)
            shift: (bool): Whether to apply random shifts to the image. Else the image will always be centered.
            background_img (Optional[str]): Path to the background image. If None, a black background will be used.
                If the background image doesn't match the visual field size, it will be resized so one side matches and the other
                side is cropped randomly.
        """
        super().__init__(image_rescale_range)
        self.vision_width = vision_width
        self.vision_height = vision_height

    def transform(self, img: torch.Tensor, trans_factor: float) -> torch.Tensor:
        """Apply the scale and shift transformation to an input image.

        Args:
        ----
            img (torch.Tensor): The input image tensor to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            torch.Tensor: The transformed image tensor.
        """
        # Scale the image
        scaled_size = (int(img.shape[1] * trans_factor), int(img.shape[2] * trans_factor))
        img = T.Resize(scaled_size)(img)

        # Create a black background
        background = torch.zeros(
            (img.shape[0], self.vision_height, self.vision_width), dtype=img.dtype
        )

        # Center position in the target area
        center_x = self.vision_width // 2
        center_y = self.vision_height // 2

        # Calculate the initial top-left position to center the image
        initial_y = center_y - (scaled_size[0] // 2)
        initial_x = center_x - (scaled_size[1] // 2)

        # Calculate maximum possible shifts to keep the image within the bounds
        max_x_shift = min(initial_x, self.vision_width - (initial_x + scaled_size[0]))
        max_y_shift = min(initial_y, self.vision_height - (initial_y + scaled_size[1]))

        # Random shift within the calculated range
        x_shift = np.random.randint(-max_x_shift, max_x_shift + 1)
        y_shift = np.random.randint(-max_y_shift, max_y_shift + 1)

        # Calculate the final position with shift
        final_x = initial_x + x_shift
        final_y = initial_y + y_shift

        # Paste the scaled image onto the background
        background[
            :, final_y : final_y + img.shape[1], final_x : final_x + img.shape[2]
        ] = img

        return background


class ShotNoiseTransform(ContinuousTransform):
    """Apply random shot noise to the input image."""

    def __init__(self, lambda_range: Tuple[float, float]) -> None:
        """Initialize the ShotNoiseTransform.

        Args:
        ----
            lambda_range (Tuple[float, float]): Range of shot noise intensity factors. For an identity transform, set the range to (0, 0) to disable the shot noise.

        """
        super().__init__(lambda_range)

    def transform(self, img: torch.Tensor, trans_factor: float) -> torch.Tensor:
        """Apply shot noise to the input image.

        Args:
        ----
            img (torch.Tensor): The input image tensor to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            torch.Tensor: The transformed image tensor with added shot noise.

        """
        if trans_factor <= 0:
            return img

        # Apply shot noise
        noise = torch.poisson(img * trans_factor) / trans_factor
        noisy_img = torch.clamp(noise, 0, 1)  # Assuming img is normalized to [0, 1]

        return noisy_img


class ContrastTransform(ContinuousTransform):
    """Apply random contrast adjustment to the input image."""

    def __init__(self, contrast_range: Tuple[float, float]) -> None:
        """Initialize the ContrastTransform.

        Args:
        ----
            contrast_range (Tuple[float, float]): Range of contrast adjustment factors. For an identity transform, set the range to (1, 1).

        """
        super().__init__(contrast_range)
        self.contrast_transform = T.ColorJitter(contrast=contrast_range[1])

    def transform(self, img: torch.Tensor, trans_factor: float) -> torch.Tensor:
        """Apply random contrast adjustment to the input image.

        Args:
        ----
            img (torch.Tensor): The input image tensor to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            torch.Tensor: The transformed image tensor with adjusted contrast.

        """
        return self.contrast_transform(img)
