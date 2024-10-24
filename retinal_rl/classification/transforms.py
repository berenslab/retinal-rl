"""Provides custom transformation classes for tensor-based image processing tasks."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torchvision.transforms.functional as tf
from torch import Tensor, nn


class ContinuousTransform(nn.Module, ABC):
    """Base class for continuous tensor transformations."""

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
    def transform(self, img: Tensor, trans_factor: float) -> Tensor:
        """Apply the transformation to the input tensor.

        Args:
            img: Input tensor of shape (C, H, W)
            trans_factor: The transformation factor to apply

        Returns:
            Transformed tensor of shape (C, H, W)
        """
        raise NotImplementedError

    def forward(self, img: Tensor) -> Tensor:
        """Randomly apply the transformation to the input tensor."""
        trans_factor = (
            torch.rand(1).item() * (self.trans_range[1] - self.trans_range[0])
            + self.trans_range[0]
        )
        return self.transform(img, trans_factor)


class IlluminationTransform(ContinuousTransform):
    """Apply random illumination (brightness) adjustment to the input tensor."""

    def __init__(self, brightness_range: Tuple[float, float]) -> None:
        """Initialize the IlluminationTransform.

        Args:
            brightness_range: Range of brightness adjustment factors (1, 1) is the identity transform.
        """
        super().__init__(brightness_range)

    def transform(self, img: Tensor, trans_factor: float) -> Tensor:
        """Apply illumination adjustment to the input tensor."""
        return img * trans_factor


class BlurTransform(ContinuousTransform):
    """Apply random Gaussian blur to the input tensor."""

    def __init__(self, blur_range: Tuple[float, float]) -> None:
        """Initialize the BlurTransform.

        Args:
            blur_range: Range of blur radii. For identity transform, set to (0, 0).
        """
        super().__init__(blur_range)

    def transform(self, img: Tensor, trans_factor: float) -> Tensor:
        """Apply Gaussian blur to the input tensor."""
        if trans_factor == 0:
            return img

        # Calculate kernel size based on sigma (trans_factor)
        kernel_size = max(3, int(2 * round(trans_factor * 3) + 1))
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

        return tf.gaussian_blur(
            img,
            kernel_size=[kernel_size, kernel_size],
            sigma=[trans_factor, trans_factor],
        )


class ScaleShiftTransform(ContinuousTransform):
    """Apply random scale and shift transformations to the input tensor. The scale is controlled by the transformation factor, while the shift is random across the entire canvas."""

    def __init__(
        self,
        vision_width: int,
        vision_height: int,
        image_rescale_range: Tuple[float, float],
    ) -> None:
        """Initialize the ScaleShiftTransform.

        Args:
            vision_width: Width of the visual field
            vision_height: Height of the visual field
            image_rescale_range: Range of image rescaling factors. (1, 1) is the identity transform.
        """
        super().__init__(image_rescale_range)
        self.vision_width = vision_width
        self.vision_height = vision_height

    def transform(self, img: Tensor, trans_factor: float) -> Tensor:
        """Apply scale and shift transformation to the input tensor."""
        # Create output tensor of target size
        output = torch.zeros(
            img.size(0),
            self.vision_height,
            self.vision_width,
            device=img.device,
            dtype=img.dtype,
        )

        # Calculate scaled dimensions
        _, h, w = img.shape
        new_h = int(h * trans_factor)
        new_w = int(w * trans_factor)

        # Resize image using TF.resize
        scaled_img = tf.resize(
            img, size=[new_h, new_w], antialias=True
        )  # Calculate initial position to center the image

        start_y = (self.vision_height - new_h) // 2
        start_x = (self.vision_width - new_w) // 2

        # Calculate maximum possible shifts
        max_y_shift = min(start_y, self.vision_height - (start_y + new_h))
        max_x_shift = min(start_x, self.vision_width - (start_x + new_w))

        # Apply random shift
        y_shift = torch.randint(-max_y_shift, max_y_shift + 1, (1,)).item()
        x_shift = torch.randint(-max_x_shift, max_x_shift + 1, (1,)).item()

        # Calculate final position
        final_y = start_y + y_shift
        final_x = start_x + x_shift

        # Place the scaled image in the output tensor
        output[:, final_y : final_y + new_h, final_x : final_x + new_w] = scaled_img

        return output


class ShotNoiseTransform(ContinuousTransform):
    """Apply random shot noise to the input tensor."""

    def __init__(self, lambda_range: Tuple[float, float]) -> None:
        """Initialize the ShotNoiseTransform.

        Args:
            lambda_range: Range of shot noise intensity factors. (0, 0) is the identity transform.
        """
        super().__init__(lambda_range)

    def transform(self, img: Tensor, trans_factor: float) -> Tensor:
        """Apply shot noise to the input tensor."""
        if trans_factor <= 0:
            return img

        # Apply shot noise using Poisson distribution
        noisy = torch.poisson(img * trans_factor) / trans_factor
        return torch.clamp(noisy, 0, 1)


class ContrastTransform(ContinuousTransform):
    """Apply random contrast adjustment to the input tensor."""

    def __init__(self, contrast_range: Tuple[float, float]) -> None:
        """Initialize the ContrastTransform.

        Args:
            contrast_range: Range of contrast adjustment factors.
                          For identity transform, set to (1, 1).
        """
        super().__init__(contrast_range)

    def transform(self, img: Tensor, trans_factor: float) -> Tensor:
        """Apply contrast adjustment to the input tensor."""
        mean = torch.mean(img, dim=(-2, -1), keepdim=True)
        return (img - mean) * trans_factor + mean
