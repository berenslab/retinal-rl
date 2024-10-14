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

import numpy as np
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter


class ContinuousTransform(nn.Module, ABC):
    """Base class for continuous image transformations."""

    def __init__(self, range: Tuple[float, float]) -> None:
        """Initialize the ContinuousTransform."""
        super().__init__()
        self.range = range

    @abstractmethod
    def transform(self, img: Image.Image, trans_factor: float) -> Image.Image:
        """Apply the transformation to the input image.

        Args:
        ----
            img (Image.Image): The input PIL Image to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            Image.Image: The transformed PIL Image.

        """
        raise NotImplementedError

    def forward(self, img: Image.Image) -> Image.Image:
        """Randomly apply the transformation to the input image.

        Args:
        ----
            img (Image.Image): The input PIL Image to transform.

        Returns:
        -------
            Image.Image: The transformed PIL Image.

        """
        trans_factor = np.random.uniform(self.range[0], self.range[1])
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

    def transform(self, img: Image.Image, trans_factor: float) -> Image.Image:
        """Apply random illumination (brightness) adjustment to the input image.

        Args:
        ----
            img (Image.Image): The input PIL Image to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            Image.Image: The transformed PIL Image with adjusted illumination.

        """
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(trans_factor)


class BlurTransform(ContinuousTransform):
    """Apply random Gaussian blur to the input image."""

    def __init__(self, blur_range: Tuple[float, float]) -> None:
        """Initialize the BlurTransform.

        Args:
        ----
            blur_range (Tuple[float, float]): Range of blur radii. For an identity transform, set the range to (0, 0).

        """
        super().__init__(blur_range)

    def transform(self, img: Image.Image, trans_factor: float) -> Image.Image:
        """Apply random Gaussian blur to the input image.

        Args:
        ----
            img (Image.Image): The input PIL Image to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            Image.Image: The transformed PIL Image with applied blur.

        """
        return img.filter(ImageFilter.GaussianBlur(radius=trans_factor))


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
            image_rescale_range (List[float]): Range of image rescaling factors. For an identity transform, set the range to [1, 1].

        """
        super().__init__(image_rescale_range)
        self.vision_width = vision_width
        self.vision_height = vision_height

    def transform(self, img: Image.Image, trans_factor: float) -> Image.Image:
        """Apply the scale and shift transformation to an input image.

        Args:
        ----
            img (Image.Image): The input PIL Image to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            Image.Image: The transformed PIL Image.

        """
        # Scale the image
        visual_field = (self.vision_width, self.vision_height)

        scaled_size = (int(img.size[0] * trans_factor), int(img.size[1] * trans_factor))
        img = img.resize(scaled_size, Image.LANCZOS)  # type: ignore

        # Create a black background
        background = Image.new("RGB", visual_field, (0, 0, 0))

        # Center position in the target area
        center_x = visual_field[0] // 2
        center_y = visual_field[1] // 2

        # Calculate the initial top-left position to center the image
        initial_x = center_x - (scaled_size[0] // 2)
        initial_y = center_y - (scaled_size[1] // 2)

        # Calculate maximum possible shifts to keep the image within the bounds
        max_x_shift = min(initial_x, visual_field[0] - (initial_x + scaled_size[0]))
        max_y_shift = min(initial_y, visual_field[1] - (initial_y + scaled_size[1]))

        # Random shift within the calculated range
        x_shift = np.random.randint(-max_x_shift, max_x_shift + 1)
        y_shift = np.random.randint(-max_y_shift, max_y_shift + 1)

        # Calculate the final position with shift
        final_x = initial_x + x_shift
        final_y = initial_y + y_shift

        # Paste the scaled image onto the background
        background.paste(img, (final_x, final_y))

        return background


class ShotNoiseTransform(ContinuousTransform):
    """Apply random shot noise to the input image."""

    def __init__(self, lambda_range: Tuple[float, float]) -> None:
        """Initialize the ShotNoiseTransform.

        Args:
        ----
            lambda_range (Tuple[float, float]): Range of shot noise intensity factors. For an identity transform, set the range to (1, 1).

        """
        super().__init__(lambda_range)

    def transform(self, img: Image.Image, trans_factor: float) -> Image.Image:
        """Apply shot noise to the input image.

        Args:
        ----
            img (Image.Image): The input PIL Image to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            Image.Image: The transformed PIL Image with added shot noise.

        """
        # Convert PIL Image to numpy array
        img_array = np.array(img)

        # Apply shot noise
        noise = np.random.poisson(img_array * trans_factor) / trans_factor
        noisy_img_array = np.clip(noise, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(noisy_img_array)


class ContrastTransform(ContinuousTransform):
    """Apply random contrast adjustment to the input image."""

    def __init__(self, contrast_range: Tuple[float, float]) -> None:
        """Initialize the ContrastTransform.

        Args:
        ----
            contrast_range (Tuple[float, float]): Range of contrast adjustment factors. For an identity transform, set the range to (1, 1).

        """
        super().__init__(contrast_range)

    def transform(self, img: Image.Image, trans_factor: float) -> Image.Image:
        """Apply random contrast adjustment to the input image.

        Args:
        ----
            img (Image.Image): The input PIL Image to transform.
            trans_factor (float): The transformation factor to apply.

        Returns:
        -------
            Image.Image: The transformed PIL Image with adjusted contrast.

        """
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(trans_factor)
