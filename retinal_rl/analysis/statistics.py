"""Functions for analysis and statistics on a Brain model."""

import logging
from typing import Dict, List, Tuple, cast

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from retinal_rl.dataset import Imageset, ImageSubset
from retinal_rl.datasets.transforms import ContinuousTransform
from retinal_rl.models.brain import Brain, get_cnn_circuit
from retinal_rl.util import (
    FloatArray,
    is_nonlinearity,
    rf_size_and_start,
)

logger = logging.getLogger(__name__)


def transform_base_images(
    imageset: Imageset, num_steps: int, num_images: int
) -> Dict[str, Dict[str, Dict[float, List[Tensor]]]]:
    """Apply transformations to a set of images from an Imageset."""
    images: List[Image.Image] = []

    base_dataset = imageset.base_dataset
    base_len = imageset.base_len

    for _ in range(num_images):
        src, _ = base_dataset[np.random.randint(base_len)]
        images.append(src)

    results: Dict[str, Dict[str, Dict[float, List[Tensor]]]] = {
        "source_transforms": {},
        "noise_transforms": {},
    }

    transforms: List[Tuple[str, nn.Module]] = []
    transforms += [
        ("source_transforms", transform) for transform in imageset.source_transforms
    ]
    transforms += [
        ("noise_transforms", transform) for transform in imageset.noise_transforms
    ]

    for category, transform in transforms:
        if isinstance(transform, ContinuousTransform):
            results[category][transform.name] = {}
            trans_range: Tuple[float, float] = transform.trans_range
            transform_steps = np.linspace(*trans_range, num_steps)
            for step in transform_steps:
                results[category][transform.name][step] = []
                for img in images:
                    results[category][transform.name][step].append(
                        imageset.to_tensor(transform.transform(img, step))
                    )

    return results


def reconstruct_images(
    device: torch.device,
    brain: Brain,
    decoder: str,
    test_set: Imageset,
    train_set: Imageset,
    sample_size: int,
) -> Dict[str, List[Tuple[Tensor, int]]]:
    """Compute reconstructions of a set of training and test images using a Brain model."""
    brain.eval()  # Set the model to evaluation mode

    def collect_reconstructions(
        dataset: Imageset, sample_size: int
    ) -> Tuple[
        List[Tuple[Tensor, int]], List[Tuple[Tensor, int]], List[Tuple[Tensor, int]]
    ]:
        """Collect reconstructions for a subset of a dataset."""
        source_subset: List[Tuple[Tensor, int]] = []
        input_subset: List[Tuple[Tensor, int]] = []
        estimates: List[Tuple[Tensor, int]] = []
        indices = torch.randperm(len(dataset))[:sample_size]

        with torch.no_grad():  # Disable gradient computation
            for index in indices:
                src, img, k = dataset[int(index)]
                src = src.to(device)
                img = img.to(device)
                stimulus = {"vision": img.unsqueeze(0)}
                response = brain(stimulus)
                rec_img = response[decoder].squeeze(0)
                pred_k = response["classifier"].argmax().item()
                source_subset.append((src.cpu(), k))
                input_subset.append((img.cpu(), k))
                estimates.append((rec_img.cpu(), pred_k))

        return source_subset, input_subset, estimates

    train_source, train_input, train_estimates = collect_reconstructions(
        train_set, sample_size
    )
    test_source, test_input, test_estimates = collect_reconstructions(
        test_set, sample_size
    )

    return {
        "train_sources": train_source,
        "train_inputs": train_input,
        "train_estimates": train_estimates,
        "test_sources": test_source,
        "test_inputs": test_input,
        "test_estimates": test_estimates,
    }


def cnn_statistics(
    device: torch.device,
    dataset: Imageset,
    brain: Brain,
    max_sample_size: int = 0,
) -> Dict[str, Dict[str, FloatArray]]:
    """Compute statistics for a convolutional encoder model.

    This function analyzes the input data and each layer of the convolutional encoder,
    computing various statistical measures and properties.

    Args:
    ----
        device (torch.device): The device to run computations on.
        dataset (Imageset): The dataset to analyze.
        brain (Brain): The trained Brain model.
        max_sample_size (int, optional): Maximum number of samples to use. If 0, use all samples. Defaults to 0.

    Returns:
    -------
        Dict[str, Dict[str, FloatArray]]: A nested dictionary containing statistics for the input and each layer.
        The outer dictionary is keyed by layer names (with "input" for the input data), and each inner dictionary
        contains the following keys:

        - "receptive_fields": FloatArray of shape (out_channels, in_channels, height, width)
            Receptive fields for the layer (identity matrix for input).
        - "pixel_histograms": FloatArray of shape (num_channels, num_bins)
            Histograms of pixel values for each channel.
        - "histogram_bin_edges": FloatArray of shape (num_bins + 1,)
            Bin edges for the pixel histograms.
        - "mean_power_spectrum": FloatArray of shape (height, width)
            Mean power spectrum across all channels.
        - "var_power_spectrum": FloatArray of shape (height, width)
            Variance of the power spectrum across all channels.
        - "mean_autocorr": FloatArray of shape (height, width)
            Mean autocorrelation across all channels.
        - "var_autocorr": FloatArray of shape (height, width)
            Variance of the autocorrelation across all channels.
        - "num_channels": FloatArray of shape (1,)
            Number of channels in the layer.

    Note:
    ----
        The function only analyzes the longest sequence of convolutional layers it can find.
        If max_sample_size is specified and smaller than the dataset size, a random subset of the data is used.

    """
    # Get the input shape and the CNN layers
    brain.eval()
    brain.to(device)
    input_shape, cnn_layers = get_cnn_circuit(brain)
    nclrs, hght, wdth = input_shape

    # Prepare subsample
    original_size = len(dataset)
    logger.info(f"Original dataset size: {original_size}")

    if max_sample_size > 0 and original_size > max_sample_size:
        indices: List[int] = torch.randperm(original_size)[:max_sample_size].tolist()
        subset = ImageSubset(dataset, indices=indices)
        logger.info(f"Reducing dataset size for cnn_statistics to {max_sample_size}")
    else:
        indices = list(range(original_size))
        subset = ImageSubset(dataset, indices=indices)
        logger.info("Using full dataset for cnn_statistics")

    dataloader = DataLoader(subset, batch_size=64, shuffle=False)

    # Initialize results dictionary
    results: Dict[str, Dict[str, FloatArray]] = {}

    # Analyze input statistics
    input_spectral = _layer_spectral_analysis(device, dataloader, nn.Identity())
    input_histograms = _layer_pixel_histograms(device, dataloader, nn.Identity())
    # num channels as FloatArray

    # Load input statistics into results dictionary
    results["input"] = {
        "receptive_fields": np.eye(nclrs)[
            :, :, np.newaxis, np.newaxis
        ],  # Identity for input
        "shape": np.array(input_shape, dtype=np.float64),
        "pixel_histograms": input_histograms["channel_histograms"],
        "histogram_bin_edges": input_histograms["bin_edges"],
        "mean_power_spectrum": input_spectral["mean_power_spectrum"],
        "var_power_spectrum": input_spectral["var_power_spectrum"],
        "mean_autocorr": input_spectral["mean_autocorr"],
        "var_autocorr": input_spectral["var_autocorr"],
        "num_channels": np.array(nclrs, dtype=np.float64),
    }

    # Initialize loop variables
    imgsz = [1, nclrs, hght, wdth]
    obs = torch.zeros(size=imgsz, device=device, requires_grad=True)
    hmn: int = 0
    head_layers: List[nn.Module] = []

    # Analyze each layer
    for layer_name, layer in cnn_layers.items():
        head_layers.append(layer)
        # Sequential model up to the current layer
        head_model = nn.Sequential(*head_layers)

        x = head_model(obs)

        if is_nonlinearity(layer):
            continue  # Skip layers without out_channels (e.g., activation functions)

        ochns: int

        if isinstance(layer, nn.Conv2d):
            ochns: int = layer.out_channels
        else:
            raise NotImplementedError(
                "Can only compute receptive fields for 2d convolutional layers"
            )

        # Compute receptive fields
        hsz, wsz = x.shape[2:]  # Get the current height and width

        hidx = (hsz - 1) // 2
        widx = (wsz - 1) // 2

        hrf_size, wrf_size, hmn, wmn = rf_size_and_start(head_layers, hidx, widx)
        grads: List[Tensor] = []

        for j in range(ochns):
            grad = torch.autograd.grad(x[0, j, hidx, widx], obs, retain_graph=True)[0]
            grads.append(grad[0, :, hmn : hmn + hrf_size, wmn : wmn + wrf_size])
        lrfs = torch.stack(grads).cpu().numpy()

        # Perform spectral analysis and histogram computation on the transformed data
        layer_spectral = _layer_spectral_analysis(device, dataloader, head_model)
        layer_histograms = _layer_pixel_histograms(device, dataloader, head_model)

        results[layer_name] = {
            "receptive_fields": lrfs,
            "pixel_histograms": layer_histograms["channel_histograms"],
            "histogram_bin_edges": layer_histograms["bin_edges"],
            "mean_power_spectrum": layer_spectral["mean_power_spectrum"],
            "var_power_spectrum": layer_spectral["var_power_spectrum"],
            "mean_autocorr": layer_spectral["mean_autocorr"],
            "var_autocorr": layer_spectral["var_autocorr"],
            "num_channels": np.array(ochns, dtype=np.float64),
        }

    return results


def _layer_pixel_histograms(
    device: torch.device,
    dataloader: DataLoader[Tuple[Tensor, Tensor, int]],
    model: nn.Module,
    num_bins: int = 20,
) -> Dict[str, FloatArray]:
    """Compute histograms of pixel/activation values for each channel across all data in a dataset."""
    _, first_batch, _ = next(iter(dataloader))
    with torch.no_grad():
        first_batch = model(first_batch.to(device))
    num_channels: int = first_batch.shape[1]

    # Initialize variables for dynamic range computation
    global_min = torch.full((num_channels,), float("inf"), device=device)
    global_max = torch.full((num_channels,), float("-inf"), device=device)

    # First pass: compute global min and max
    total_elements = 0

    for _, batch, _ in dataloader:
        with torch.no_grad():
            batch = model(batch.to(device))
        batch_min, _ = batch.view(-1, num_channels).min(dim=0)
        batch_max, _ = batch.view(-1, num_channels).max(dim=0)
        global_min = torch.min(global_min, batch_min)
        global_max = torch.max(global_max, batch_max)
        total_elements += batch.numel() // num_channels

    # Compute histogram parameters
    hist_range: tuple[float, float] = (global_min.min().item(), global_max.max().item())

    histograms: Tensor = torch.zeros(
        (num_channels, num_bins), dtype=torch.float64, device=device
    )

    for _, batch, _ in dataloader:
        with torch.no_grad():
            batch = model(batch.to(device))
        for c in range(num_channels):
            channel_data = batch[:, c, :, :].reshape(-1)
            hist = torch.histc(
                channel_data, bins=num_bins, min=hist_range[0], max=hist_range[1]
            )
            histograms[c] += hist

    bin_width = (hist_range[1] - hist_range[0]) / num_bins
    normalized_histograms = histograms / (total_elements * bin_width / num_channels)

    return {
        "bin_edges": np.linspace(
            hist_range[0], hist_range[1], num_bins + 1, dtype=np.float64
        ),
        "channel_histograms": normalized_histograms.cpu().numpy(),
    }


def _layer_spectral_analysis(
    device: torch.device,
    dataloader: DataLoader[Tuple[Tensor, Tensor, int]],
    model: nn.Module,
) -> Dict[str, FloatArray]:
    """Compute spectral analysis statistics for each channel across all data in a dataset."""
    _, first_batch, _ = next(iter(dataloader))
    with torch.no_grad():
        first_batch = model(first_batch.to(device))
    image_size = first_batch.shape[1:]

    # Initialize variables for dynamic range computation
    mean_power_spectrum = torch.zeros(image_size, dtype=torch.float64, device=device)
    m2_power_spectrum = torch.zeros(image_size, dtype=torch.float64, device=device)
    mean_autocorr = torch.zeros(image_size, dtype=torch.float64, device=device)
    m2_autocorr = torch.zeros(image_size, dtype=torch.float64, device=device)
    autocorr: Tensor = torch.zeros(image_size, dtype=torch.float64, device=device)

    count = 0

    for _, batch, _ in dataloader:
        with torch.no_grad():
            batch = model(batch.to(device))
        for image in batch:
            count += 1

            # Compute power spectrum
            power_spectrum = torch.abs(fft.fft2(image)) ** 2

            # Compute power spectrum statistics
            mean_power_spectrum += power_spectrum
            m2_power_spectrum += power_spectrum**2

            # Compute normalized autocorrelation
            autocorr = cast(Tensor, fft.ifft2(power_spectrum)).real
            max_abs_autocorr = torch.amax(torch.abs(autocorr), dim=(-2, -1), keepdim=True)
            autocorr = autocorr / (max_abs_autocorr + 1e-8)

            # Compute autocorrelation statistics
            mean_autocorr += autocorr
            m2_autocorr += autocorr**2

    mean_power_spectrum /= count
    mean_autocorr /= count
    var_power_spectrum = m2_power_spectrum / count - (mean_power_spectrum / count) ** 2
    var_autocorr = m2_autocorr / count - (mean_autocorr / count) ** 2

    return {
        "mean_power_spectrum": mean_power_spectrum.cpu().numpy(),
        "var_power_spectrum": var_power_spectrum.cpu().numpy(),
        "mean_autocorr": mean_autocorr.cpu().numpy(),
        "var_autocorr": var_autocorr.cpu().numpy(),
    }
