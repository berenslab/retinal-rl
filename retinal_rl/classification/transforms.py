"""Provides custom transformation classes for image processing tasks, applicable to both classification and reinforcement learning scenarios.

It includes various image transformations:
- ScaleShiftTransform
- ShotNoiseTransform
- ContrastTransform
- IlluminationTransform
- BlurTransform
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn


class ContinuousTransform(nn.Module, ABC):
    """Base class for continuous image transformations."""

    def __init__(self, trans_range: Tuple[float, float]) -> None:
        """Initialize the ContinuousTransform."""
        super().__init__()
        self.trans_range: Tuple[float, float] = trans_range

    @property
    def name(self) -> str:
        """Return  a pretty name of the transformation."""
        name = self.__class__.__name__
        # Remove the "Transform" suffix
        name = name.replace("Transform", "")
        # decamelcase
        return name.replace("([a-z])([A-Z])", r"\1 \2").lower()

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
        shift: bool = True,
        background_img: Optional[str] = None,
    ) -> None:
        """Initialize the ScaleShiftTransform.

        Args:
        ----
            vision_width (int): The width of the visual field.
            vision_height (int): The height of the visual field.
            image_rescale_range (List[float]): Range of image rescaling factors. For an identity transform, set the range to [1, 1].
            shift: (bool): Whether to apply random shifts to the image. Else the image will always be centered.
            background_img (Optional[str]): Path to the background image. If None, a black background will be used.
                If the background image doesn't match the visual field size, it will be resized so one side matches and the other
                side is cropped randomly.

        """
        super().__init__(image_rescale_range)
        self.visual_field = (vision_width, vision_height)
        self.shift = shift
        if background_img is not None:
            self.background = Image.open(background_img).convert("RGB")
            bg_ratio = self.background.size[0] / self.background.size[1]
            vf_ratio = self.visual_field[0] / self.visual_field[1]
            if bg_ratio > vf_ratio:
                # Background is wider than visual field
                new_height = self.visual_field[1]
                new_width = int(new_height * bg_ratio)
            else:
                # Background is taller than visual field
                new_width = self.visual_field[0]
                new_height = int(new_width / bg_ratio)
            self.background = self.background.resize((new_width, new_height))
        else:
            self.background = Image.new("RGB", (self.visual_field[0], self.visual_field[1]), (0, 0, 0))


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

        scaled_size = (int(img.size[0] * trans_factor), int(img.size[1] * trans_factor))
        img = img.resize(scaled_size, Image.LANCZOS)  # type: ignore

        # Center position in the target area
        center_x = self.visual_field[0] // 2
        center_y = self.visual_field[1] // 2

        # Calculate the initial top-left position to center the image
        pos_x = center_x - (scaled_size[0] // 2)
        pos_y = center_y - (scaled_size[1] // 2)

        if self.shift:
            # Calculate maximum possible shifts to keep the image within the bounds
            max_x_shift = min(pos_x, self.visual_field[0] - (pos_x + scaled_size[0]))
            max_y_shift = min(pos_y, self.visual_field[1] - (pos_y + scaled_size[1]))

            # Random shift within the calculated range
            x_shift = np.random.randint(-max_x_shift, max_x_shift + 1)
            y_shift = np.random.randint(-max_y_shift, max_y_shift + 1)

            # Calculate the final position with shift
            pos_x = pos_x + x_shift
            pos_y = pos_y + y_shift


        # get random crop of background image matching the visual field size
        background = self.background.copy()
        if self.background.size != self.visual_field:
            width_diff = self.background.size[0] - self.visual_field[0]
            x_start = 0 if width_diff == 0 else np.random.randint(0, width_diff)
            height_diff = self.background.size[1] - self.visual_field[1]
            y_start = 0 if height_diff == 0 else np.random.randint(0, height_diff)
            background = background.crop((x_start, y_start, x_start + self.visual_field[0], y_start + self.visual_field[1]))
        # Paste the scaled image onto the background
        background.paste(img, (pos_x, pos_y))

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
        if trans_factor <= 0:
            return img

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
