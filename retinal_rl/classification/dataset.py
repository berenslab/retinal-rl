import logging
from typing import List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ScaleShiftTransform:
    def __init__(
        self,
        visual_field: List[int],
        scale_range: List[float],
    ) -> None:
        self.target_size = visual_field
        self.scale_range = scale_range

    def __call__(self, img: Image.Image) -> Image.Image:
        # Scale the image
        visual_field = tuple(self.target_size)
        scale_range = tuple(self.scale_range)

        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        scaled_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
        img = img.resize(scaled_size, Image.LANCZOS)

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

        # Log warnings if shifts go out of bounds
        if (
            final_x < 0
            or final_y < 0
            or final_x > visual_field[0] - scaled_size[0]
            or final_y > visual_field[1] - scaled_size[1]
        ):
            logger.warning(
                f"Shift out of bounds adjusted: final_x={final_x}, final_y={final_y}, "
                f"initial_x+shift={initial_x + x_shift}, initial_y+shift={initial_y + y_shift}."
            )

        # Paste the scaled image onto the background
        background.paste(img, (final_x, final_y))

        return background
