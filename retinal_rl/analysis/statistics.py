import logging
from typing import Dict, Iterator, List, Tuple, cast

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
from captum.attr import NeuronGradient
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from retinal_rl.models.brain import Brain
from retinal_rl.models.circuits.convolutional import ConvolutionalEncoder
from retinal_rl.util import FloatArray, encoder_out_size, is_activation, rf_size_and_start

logger = logging.getLogger(__name__)


def reconstruct_images(
    device: torch.device,
    brain: Brain,
    test_set: Dataset[Tuple[Tensor, int]],
    train_set: Dataset[Tuple[Tensor, int]],
    sample_size: int,
) -> Dict[str, List[Tuple[Tensor, int]]]:
    brain.eval()  # Set the model to evaluation mode

    def collect_reconstructions(
        data_set: Dataset[Tuple[Tensor, int]], sample_size: int
    ) -> Tuple[List[Tuple[Tensor, int]], List[Tuple[Tensor, int]]]:
        subset: List[Tuple[Tensor, int]] = []
        estimates: List[Tuple[Tensor, int]] = []
        indices = torch.randperm(len(data_set))[:sample_size]

        with torch.no_grad():  # Disable gradient computation
            for index in indices:
                img, k = data_set[index]
                img = img.to(device)
                stimulus = {"vision": img.unsqueeze(0)}
                response = brain(stimulus)
                rec_img = response["decoder"].squeeze(0)
                pred_k = response["classifier"].argmax().item()
                subset.append((img.cpu(), k))
                estimates.append((rec_img.cpu(), pred_k))

        return subset, estimates

    train_subset, train_estimates = collect_reconstructions(train_set, sample_size)
    test_subset, test_estimates = collect_reconstructions(test_set, sample_size)

    return {
        "train_subset": train_subset,
        "train_estimates": train_estimates,
        "test_subset": test_subset,
        "test_estimates": test_estimates,
    }


def cnn_linear_receptive_fields(
    device: torch.device, enc: ConvolutionalEncoder
) -> Dict[str, FloatArray]:
    """Return the receptive fields of every layer of a convnet as computed by neural gradients.

    The returned dictionary is indexed by layer name and the shape of the array
    (#layer_channels, #input_channels, #rf_height, #rf_width).
    """
    enc.eval()
    nclrs, hght, wdth = enc.input_shape
    ochns = nclrs

    imgsz = [1, nclrs, hght, wdth]
    # Obs requires grad
    obs = torch.zeros(size=imgsz, device=device, requires_grad=True)

    stas: Dict[str, FloatArray] = {}

    mdls: List[nn.Module] = []

    with torch.no_grad():
        for lyrnm, mdl in enc.conv_head.named_children():
            gradient_calculator = NeuronGradient(enc, mdl)
            mdls.append(mdl)

            # check if mdl has out channels
            if hasattr(mdl, "out_channels"):
                ochns = mdl.out_channels
            hsz, wsz = encoder_out_size(mdls, hght, wdth)

            hidx = (hsz - 1) // 2
            widx = (wsz - 1) // 2

            hrf_size, wrf_size, hmn, wmn = rf_size_and_start(mdls, hidx, widx)

            hmx = hmn + hrf_size
            wmx = wmn + wrf_size

            stas[lyrnm] = np.zeros((ochns, nclrs, hrf_size, wrf_size))

            for j in range(ochns):
                grad = (
                    gradient_calculator.attribute(obs, (j, hidx, widx))[
                        0, :, hmn:hmx, wmn:wmx
                    ]
                    .cpu()
                    .numpy()
                )

                stas[lyrnm][j] = grad

    return stas


def layer_pixel_histograms(
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


def layer_spectral_analysis(
    device: torch.device,
    dataloader: DataLoader[Tuple[Tensor, int]],
) -> Dict[str, FloatArray]:
    """Compute power spectrum statistics for each channel across all data in a dataset."""
    first_batch, _ = next(iter(dataloader))
    first_batch = first_batch.to(device)
    image_size = (
        first_batch.shape[1],
        first_batch.shape[2],
        first_batch.shape[3],
    )

    # Initialize variables for dynamic range computation
    mean_power_spectrum = torch.zeros(image_size, dtype=torch.float64, device=device)
    m2_power_spectrum = torch.zeros(image_size, dtype=torch.float64, device=device)
    mean_autocorr = torch.zeros(image_size, dtype=torch.float64, device=device)
    m2_autocorr = torch.zeros(image_size, dtype=torch.float64, device=device)
    autocorr: Tensor = torch.zeros(image_size, dtype=torch.float64, device=device)

    batch_count = 0

    for batch, _ in dataloader:
        batch = batch.to(device)
        batch_count += 1

        # Compute power spectrum
        power_spectrum = torch.abs(fft.fft2(batch)) ** 2

        # Compute power spectrum statistics
        delta_power = power_spectrum - mean_power_spectrum
        mean_power_spectrum += delta_power.mean(dim=0)
        delta2_power = power_spectrum - mean_power_spectrum
        m2_power_spectrum += (delta_power * delta2_power).mean(dim=0)

        # Compute autocorrelation
        autocorr = cast(Tensor, fft.ifft2(power_spectrum)).real
        max_abs_autocorr = torch.max(torch.abs(autocorr))
        autocorr = autocorr / max_abs_autocorr

        # Compute autocorrelation statistics
        delta_autocorr = autocorr - mean_autocorr
        mean_autocorr += delta_autocorr.mean(dim=0)
        delta2_autocorr = autocorr - mean_autocorr
        m2_autocorr += (delta_autocorr * delta2_autocorr).mean(dim=0)

    var_power_spectrum = m2_power_spectrum / batch_count
    var_autocorr = m2_autocorr / batch_count

    return {
        "mean_power_spectrum": mean_power_spectrum.cpu().numpy(),
        "var_power_spectrum": var_power_spectrum.cpu().numpy(),
        "mean_autocorr": mean_autocorr.cpu().numpy(),
        "var_autocorr": var_autocorr.cpu().numpy(),
    }


def cnn_statistics(
    device: torch.device,
    dataset: Dataset[Tuple[Tensor, int]],
    encoder: ConvolutionalEncoder,
    max_sample_size: int = 0,
) -> Dict[str, Dict[str, FloatArray]]:
    encoder.eval()

    # Set sample size
    original_size = len(dataset)
    logger.info(f"Original dataset size: {original_size}")

    if max_sample_size > 0 and original_size > max_sample_size:
        indices: List[int] = torch.randperm(original_size)[:max_sample_size].tolist()
        dataset = Subset(dataset, indices=indices)
        logger.info(f"Reducing dataset size for cnn_statistics to {max_sample_size}")
    else:
        logger.info("Using full dataset for cnn_statistics")

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    results: Dict[str, Dict[str, FloatArray]] = {}

    # Compute receptive fields for all layers
    receptive_fields = cnn_linear_receptive_fields(device, encoder)

    # Analyze input data
    input_spectral = layer_spectral_analysis(device, dataloader)
    input_histograms = layer_pixel_histograms(device, dataloader)
    # num channels as FloatArray
    num_channels = encoder.input_shape[0]
    num_channels = np.array(num_channels, dtype=np)

    results["input"] = {
        "receptive_fields": np.eye(3)[:, :, np.newaxis, np.newaxis],  # Identity for input
        "pixel_histograms": input_histograms["channel_histograms"],
        "histogram_bin_edges": input_histograms["bin_edges"],
        "mean_power_spectrum": input_spectral["mean_power_spectrum"],
        "var_power_spectrum": input_spectral["var_power_spectrum"],
        "mean_autocorr": input_spectral["mean_autocorr"],
        "var_autocorr": input_spectral["var_autocorr"],
        "num_channels": num_channels,
    }

    # Analyze each layer
    for name, layer in encoder.conv_head.named_children():
        if hasattr(layer, "out_channels"):
            num_channels = layer.out_channels

        # Only analyze activation layers
        if not is_activation(layer):
            continue

        # Create a temporary model up to the current layer
        temp_model = nn.Sequential(
            *list(encoder.conv_head.children())[
                : list(encoder.conv_head.named_children()).index((name, layer)) + 1
            ]
        )

        # Create a new dataloader that applies the temporary model to the data
        transformed_dataloader = _TransformedDataLoader(device, dataloader, temp_model)

        # Perform spectral analysis and histogram computation on the transformed data
        layer_spectral = layer_spectral_analysis(device, transformed_dataloader)
        layer_histograms = layer_pixel_histograms(device, transformed_dataloader)

        results[name] = {
            "receptive_fields": receptive_fields[name],
            "pixel_histograms": layer_histograms["channel_histograms"],
            "histogram_bin_edges": layer_histograms["bin_edges"],
            "mean_power_spectrum": layer_spectral["mean_power_spectrum"],
            "var_power_spectrum": layer_spectral["var_power_spectrum"],
            "mean_autocorr": layer_spectral["mean_autocorr"],
            "var_autocorr": layer_spectral["var_autocorr"],
            "num_channels": np.array([num_channels], dtype=np),
        }

    return results


class _TransformedDataLoader(DataLoader[Tuple[Tensor, int]]):
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
