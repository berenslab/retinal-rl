"""Provides custom transformation classes for image processing tasks, applicable to both classification and reinforcement learning scenarios.

It includes various image transformations:
- ScaleShiftTransform
- ShotNoiseTransform
- ContrastTransform
"""

from typing import List, Tuple

import numpy as np
import torch.nn as nn
from PIL import Image, ImageEnhance


class ScaleShiftTransform(nn.Module):
    def __init__(
        self,
        vision_width: int,
        vision_height: int,
        image_rescale_range: List[float],
    ) -> None:
        super().__init__()
        self.vision_width = vision_width
        self.vision_height = vision_height
        self.image_rescale_range = image_rescale_range

    def forward(self, img: Image.Image) -> Image.Image:
        """Apply the scale and shift transformation to an input image.

        Args:
        ----
            img (Image.Image): The input PIL Image to transform.

        Returns:
        -------
            Image.Image: The transformed PIL Image.

        """
        # Scale the image
        visual_field = (self.vision_width, self.vision_height)
        scale_range = tuple(self.image_rescale_range)

        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        scaled_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
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


class ShotNoiseTransform(nn.Module):
    def __init__(self, lambda_range: Tuple[float, float]) -> None:
        super().__init__()
        self.lambda_range = lambda_range

    def forward(self, img: Image.Image) -> Image.Image:
        """Apply shot noise to the input image.

        Args:
        ----
            img (Image.Image): The input PIL Image to transform.

        Returns:
        -------
            Image.Image: The transformed PIL Image with added shot noise.

        """
        # Convert PIL Image to numpy array
        img_array = np.array(img)

        # Apply shot noise
        lambda_value = np.random.uniform(self.lambda_range[0], self.lambda_range[1])
        noise = np.random.poisson(img_array * lambda_value) / lambda_value
        noisy_img_array = np.clip(noise, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(noisy_img_array)


class ContrastTransform(nn.Module):
    def __init__(self, contrast_range: Tuple[float, float]) -> None:
        super().__init__()
        self.contrast_range = contrast_range

    def forward(self, img: Image.Image) -> Image.Image:
        """Apply random contrast adjustment to the input image.

        Args:
        ----
            img (Image.Image): The input PIL Image to transform.

        Returns:
        -------
            Image.Image: The transformed PIL Image with adjusted contrast.

        """
        contrast_factor = np.random.uniform(
            self.contrast_range[0], self.contrast_range[1]
        )
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(contrast_factor)
