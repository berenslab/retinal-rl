from typing import Dict, Iterator, Tuple

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from retinal_rl.models.circuits.convolutional import ConvolutionalEncoder

FloatArray = NDArray[np.float64]


def image_distribution_analysis(
    device: torch.device, dataloader: DataLoader[Tuple[Tensor, int]]
):
    histogram_data = compute_channel_histograms(device, dataloader)
    power_spectrum_data = compute_power_spectrum_stats(device, dataloader)

    # Combine the results if needed
    return {**histogram_data, **power_spectrum_data}


def compute_channel_histograms(
    device: torch.device, dataloader: DataLoader[Tuple[Tensor, int]], num_bins: int = 20
) -> Dict[str, FloatArray]:
    """Compute histograms of pixel/activation values for each channel across all data in a dataset."""
    first_batch, _ = next(iter(dataloader))
    first_batch = first_batch.to(device)
    num_channels: int = first_batch.shape[1]

    # Initialize variables for dynamic range computation
    global_min = torch.full((num_channels,), float("inf"), device=device)
    global_max = torch.full((num_channels,), float("-inf"), device=device)

    # First pass: compute global min and max
    total_elements = 0

    for batch, _ in dataloader:
        batch = batch.to(device)
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

    for batch, _ in dataloader:
        batch = batch.to(device)
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


def compute_power_spectrum_stats(
    device: torch.device,
    dataloader: DataLoader[Tuple[Tensor, int]],
) -> Dict[str, FloatArray]:
    """Compute power spectrum statistics for each channel across all data in a dataset."""
    first_batch, _ = next(iter(dataloader))
    first_batch = first_batch.to(device)
    num_channels, height, width = (
        first_batch.shape[1],
        first_batch.shape[2],
        first_batch.shape[3],
    )

    mean_log_power_spectrum = torch.zeros(
        (num_channels, height, width), dtype=torch.float64, device=device
    )
    m2_log_power_spectrum = torch.zeros(
        (num_channels, height, width), dtype=torch.float64, device=device
    )
    batch_count = 0

    for batch, _ in dataloader:
        batch = batch.to(device)
        batch_count += 1

        for c in range(num_channels):
            fft_batch: Tensor = fft.fft2(batch[:, c, :, :])  # type: ignore
            log_power_spectrum = torch.log1p(torch.abs(fft_batch) ** 2)

            delta = log_power_spectrum - mean_log_power_spectrum[c]
            mean_log_power_spectrum[c] += delta.mean(dim=0)
            delta2 = log_power_spectrum - mean_log_power_spectrum[c]
            m2_log_power_spectrum[c] += (delta * delta2).mean(dim=0)

    sd_log_power_spectrum = torch.sqrt(m2_log_power_spectrum / batch_count)

    x_mean_log_power_spectrum = mean_log_power_spectrum.mean(dim=2).cpu().numpy()
    y_mean_log_power_spectrum = mean_log_power_spectrum.mean(dim=1).cpu().numpy()
    x_sd_log_power_spectrum = sd_log_power_spectrum.mean(dim=2).cpu().numpy()
    y_sd_log_power_spectrum = sd_log_power_spectrum.mean(dim=1).cpu().numpy()

    return {
        "x_mean_log_power_spectrum": x_mean_log_power_spectrum,
        "y_mean_log_power_spectrum": y_mean_log_power_spectrum,
        "x_sd_log_power_spectrum": x_sd_log_power_spectrum,
        "y_sd_log_power_spectrum": y_sd_log_power_spectrum,
        # "alphas": np.array(alphas),
    }


def image_distribution_analysis_cnn(
    device: torch.device,
    dataset: Dataset[Tuple[Tensor, int]],
    encoder: ConvolutionalEncoder,
) -> Dict[str, Dict[str, FloatArray]]:
    encoder.eval()

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    results: Dict[str, Dict[str, FloatArray]] = {}

    # Analyze input images
    results["input"] = image_distribution_analysis(device, dataloader)

    for name, layer in encoder.conv_head.named_children():
        # Create a temporary model up to the current layer
        temp_model = nn.Sequential(
            *list(encoder.conv_head.children())[
                : list(encoder.conv_head.named_children()).index((name, layer)) + 1
            ]
        )

        # Create a new dataloader that applies the temporary model to the data
        transformed_dataloader = TransformedDataLoader(device, dataloader, temp_model)

        # Perform analysis on the transformed data
        layer_results: Dict[str, FloatArray] = image_distribution_analysis(
            device, transformed_dataloader
        )
        results[name] = layer_results

    return results


class TransformedDataLoader(DataLoader[Tuple[Tensor, int]]):
    def __init__(
        self,
        device: torch.device,
        dataloader: DataLoader[Tuple[Tensor, int]],
        model: nn.Module,
    ):
        self.dataloader = dataloader
        self.model = model.to(device)
        self.device = device
        super().__init__(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
        )

    def __iter__(self) -> Iterator[Tuple[Tensor, int]]:
        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                transformed_inputs = self.model(inputs)
                yield transformed_inputs, labels

    def __len__(self) -> int:
        return len(self.dataloader)
