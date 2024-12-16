from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from retinal_rl.analysis.plot import make_image_grid
from retinal_rl.classification.imageset import Imageset
from retinal_rl.classification.transforms import ContinuousTransform
from retinal_rl.util import FloatArray


@dataclass
class TransformStatistics:
    """Results of applying transformations to images."""

    source_transforms: dict[str, dict[float, list[FloatArray]]]
    noise_transforms: dict[str, dict[float, list[FloatArray]]]


def analyze(imageset: Imageset, num_steps: int, num_images: int) -> TransformStatistics:
    """Apply transformations to a set of images from an Imageset."""
    images: list[Image.Image] = []

    base_dataset = imageset.base_dataset
    base_len = imageset.base_len

    for _ in range(num_images):
        src, _ = base_dataset[np.random.randint(base_len)]
        images.append(src)

    resultss = TransformStatistics(
        source_transforms={},
        noise_transforms={},
    )

    for transforms, results in [
        (imageset.source_transforms, resultss.source_transforms),
        (imageset.noise_transforms, resultss.noise_transforms),
    ]:
        for transform in transforms:
            if isinstance(transform, ContinuousTransform):
                results[transform.name] = {}
                trans_range: tuple[float, float] = transform.trans_range
                transform_steps = np.linspace(*trans_range, num_steps)
                for step in transform_steps:
                    results[transform.name][step] = []
                    for img in images:
                        results[transform.name][step].append(
                            imageset.to_tensor(transform.transform(img, step))
                            .cpu()
                            .numpy()
                        )

    return resultss


def plot(
    source_transforms: dict[str, dict[float, list[FloatArray]]],
    noise_transforms: dict[str, dict[float, list[FloatArray]]],
) -> Figure:
    """Plot effects of source and noise transforms on images.

    Args:
        source_transforms: dictionary of source transforms (numpy arrays)
        noise_transforms: dictionary of noise transforms (numpy arrays)

    Returns:
        Figure containing the plotted transforms
    """
    num_source_transforms = len(source_transforms)
    num_noise_transforms = len(noise_transforms)
    num_transforms = num_source_transforms + num_noise_transforms
    num_images = len(
        next(iter(source_transforms.values()))[
            next(iter(next(iter(source_transforms.values())).keys()))
        ]
    )

    fig, axs = plt.subplots(num_transforms, 1, figsize=(20, 5 * num_transforms))
    if num_transforms == 1:
        axs = [axs]

    transform_index = 0

    # Plot source transforms
    for transform_name, transform_data in source_transforms.items():
        ax = axs[transform_index]
        steps = sorted(transform_data.keys())

        # Create a grid of images for each step
        images = [
            make_image_grid(
                [(img * 0.5 + 0.5) for img in transform_data[step]],
                nrow=num_images,
            )
            for step in steps
        ]
        grid = make_image_grid(images, nrow=len(steps))

        # Move channels last for imshow
        grid_display = np.transpose(grid, (1, 2, 0))
        ax.imshow(grid_display)
        ax.set_title(f"Source Transform: {transform_name}")
        ax.set_xticks(
            [(i + 0.5) * grid_display.shape[1] / len(steps) for i in range(len(steps))]
        )
        ax.set_xticklabels([f"{step:.2f}" for step in steps])
        ax.set_yticks([])

        transform_index += 1

    # Plot noise transforms
    for transform_name, transform_data in noise_transforms.items():
        ax = axs[transform_index]
        steps = sorted(transform_data.keys())

        # Create a grid of images for each step
        images = [
            make_image_grid(
                [(img * 0.5 + 0.5) for img in transform_data[step]],
                nrow=num_images,
            )
            for step in steps
        ]
        grid = make_image_grid(images, nrow=len(steps))

        # Move channels last for imshow
        grid_display = np.transpose(grid, (1, 2, 0))
        ax.imshow(grid_display)
        ax.set_title(f"Noise Transform: {transform_name}")
        ax.set_xticks(
            [(i + 0.5) * grid_display.shape[1] / len(steps) for i in range(len(steps))]
        )
        ax.set_xticklabels([f"{step:.2f}" for step in steps])
        ax.set_yticks([])

        transform_index += 1

    plt.tight_layout()
    return fig
