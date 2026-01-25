import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.utils
import torch.utils.data
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor, fft, nn
from torch.utils.data import DataLoader

from retinal_rl.analysis.plot import FigureLogger, set_integer_ticks
from retinal_rl.classification.imageset import Imageset, ImageSubset
from retinal_rl.models.brain import Brain, get_cnn_circuit
from retinal_rl.util import FloatArray, is_nonlinearity

logger = logging.getLogger(__name__)


@dataclass
class SpectralAnalysis:
    """Results of spectral analysis for a layer."""

    mean_power_spectrum: FloatArray
    var_power_spectrum: FloatArray
    mean_autocorr: FloatArray
    var_autocorr: FloatArray


@dataclass
class HistogramAnalysis:
    """Results of histogram analysis for a layer."""

    channel_histograms: FloatArray
    bin_edges: FloatArray


def spectral_analysis(
    device: torch.device,
    dataloader: DataLoader[tuple[Tensor, Tensor, int]],
    brain: Brain,
) -> dict[str, SpectralAnalysis]:
    brain.eval()
    brain.to(device)
    _, cnn_layers = get_cnn_circuit(brain)

    # Initialize results
    results: dict[str, SpectralAnalysis] = {}

    # Analyze each layer
    head_layers: list[nn.Module] = []

    for layer_name, layer in cnn_layers.items():
        head_layers.append(layer)

        if is_nonlinearity(layer):
            continue

        results[layer_name] = _layer_spectral_analysis(
            device, dataloader, nn.Sequential(*head_layers)
        )

    return results


def histogram_analysis(
    device: torch.device,
    dataloader: DataLoader[tuple[Tensor, Tensor, int]],
    brain: Brain,
) -> dict[str, HistogramAnalysis]:
    brain.eval()
    brain.to(device)
    _, cnn_layers = get_cnn_circuit(brain)

    # Initialize results
    results: dict[str, HistogramAnalysis] = {}

    # Analyze each layer
    head_layers: list[nn.Module] = []

    for layer_name, layer in cnn_layers.items():
        head_layers.append(layer)
        if is_nonlinearity(layer):
            continue
        results[layer_name] = _layer_pixel_histograms(
            device, dataloader, nn.Sequential(*head_layers)
        )

    return results


def prepare_dataset(
    imageset: Imageset, max_sample_size: int = 0
) -> DataLoader[tuple[Tensor, Tensor, int]]:
    """Prepare dataset and dataloader for analysis."""
    epoch_len = imageset.epoch_len()
    logger.info(f"Original dataset size: {epoch_len}")

    if max_sample_size > 0 and epoch_len > max_sample_size:
        indices = torch.randperm(epoch_len)[:max_sample_size].tolist()
        subset = ImageSubset(imageset, indices=indices)
        logger.info(f"Reducing dataset size for cnn_statistics to {max_sample_size}")
    else:
        indices = list(range(epoch_len))
        subset = ImageSubset(imageset, indices=indices)
        logger.info("Using full dataset for cnn_statistics")

    return DataLoader(subset, batch_size=64, shuffle=False)


def _layer_pixel_histograms(
    device: torch.device,
    dataloader: DataLoader[tuple[Tensor, Tensor, int]],
    model: nn.Module,
    num_bins: int = 20,
) -> HistogramAnalysis:
    """Compute histograms of pixel/activation values for each channel across all data in an imageset."""
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

    return HistogramAnalysis(
        normalized_histograms.cpu().numpy(),
        np.linspace(hist_range[0], hist_range[1], num_bins + 1, dtype=np.float64),
    )


def _layer_spectral_analysis(
    device: torch.device,
    dataloader: DataLoader[tuple[Tensor, Tensor, int]],
    model: nn.Module,
) -> SpectralAnalysis:
    """Compute spectral analysis statistics for each channel across all data in an imageset."""
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
            max_abs_autocorr = torch.amax(
                torch.abs(autocorr), dim=(-2, -1), keepdim=True
            )
            autocorr = autocorr / (max_abs_autocorr + 1e-8)

            # Compute autocorrelation statistics
            mean_autocorr += autocorr
            m2_autocorr += autocorr**2

    mean_power_spectrum /= count
    mean_autocorr /= count
    var_power_spectrum = m2_power_spectrum / count - (mean_power_spectrum / count) ** 2
    var_autocorr = m2_autocorr / count - (mean_autocorr / count) ** 2

    return SpectralAnalysis(
        mean_power_spectrum.cpu().numpy(),
        var_power_spectrum.cpu().numpy(),
        mean_autocorr.cpu().numpy(),
        var_autocorr.cpu().numpy(),
    )


def plot(
    log: FigureLogger,
    rf_result: dict[str, FloatArray],
    spectral_result: dict[str, SpectralAnalysis],
    histogram_result: dict[str, HistogramAnalysis],
    epoch: int,
    copy_checkpoint: bool,
):
    for layer_name, layer_rfs in rf_result.items():
        layer_spectral = spectral_result[layer_name]
        layer_histogram = histogram_result[layer_name]
        for channel in range(layer_rfs.shape[0]):
            channel_fig = layer_channel_plots(
                layer_rfs,
                layer_spectral,
                layer_histogram,
                layer_name=layer_name,
                channel=channel,
            )
            log.log_figure(
                channel_fig,
                f"{layer_name}_layer_channel_analysis",
                f"channel_{channel}",
                epoch,
                copy_checkpoint,
            )


def layer_channel_plots(
    receptive_fields: FloatArray,
    spectral: SpectralAnalysis,
    histogram: HistogramAnalysis,
    layer_name: str,
    channel: int,
) -> Figure:
    """Plot receptive fields, pixel histograms, and autocorrelation plots for a single channel in a layer."""
    axs: np.ndarray[Axes]
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"Layer: {layer_name}, Channel: {channel}", fontsize=16)

    # Receptive Fields
    rf = receptive_fields[channel]
    _plot_receptive_fields(axs[0, 0], rf)
    axs[0, 0].set_title("Receptive Field")
    axs[0, 0].set_xlabel("X")
    axs[0, 0].set_ylabel("Y")

    # Pixel Histograms
    hist = histogram.channel_histograms[channel]
    bin_edges = histogram.bin_edges
    axs[1, 0].bar(
        bin_edges[:-1],
        hist,
        width=np.diff(bin_edges),
        align="edge",
        color="gray",
        edgecolor="black",
    )
    axs[1, 0].set_title("Channel Histogram")
    axs[1, 0].set_xlabel("Value")
    axs[1, 0].set_ylabel("Empirical Probability")

    # Autocorrelation plots
    # Plot average 2D autocorrelation and variance
    
    autocorr = fft.fftshift(torch.tensor(spectral.mean_autocorr[channel])) #np shift here
    h, w = autocorr.shape
    extent = [-w // 2, w // 2, -h // 2, h // 2]
    im = axs[0, 1].imshow(
        autocorr, cmap="twilight", vmin=-1, vmax=1, origin="lower", extent=extent
    )
    axs[0, 1].set_title("Average 2D Autocorrelation")
    axs[0, 1].set_xlabel("Lag X")
    axs[0, 1].set_ylabel("Lag Y")
    fig.colorbar(im, ax=axs[0, 1])
    set_integer_ticks(axs[0, 1])

    autocorr_sd = fft.fftshift(torch.sqrt(torch.tensor(spectral.var_autocorr[channel]))) #np shift here
    im = axs[0, 2].imshow(
        autocorr_sd, cmap="inferno", origin="lower", extent=extent, vmin=0
    )
    axs[0, 2].set_title("2D Autocorrelation SD")
    axs[0, 2].set_xlabel("Lag X")
    axs[0, 2].set_ylabel("Lag Y")
    fig.colorbar(im, ax=axs[0, 2])
    set_integer_ticks(axs[0, 2])

    # Plot average 2D power spectrum
    log_power_spectrum = fft.fftshift(torch.log1p(torch.tensor(spectral.mean_power_spectrum[channel])))
    h, w = log_power_spectrum.shape

    im = axs[1, 1].imshow(
        log_power_spectrum, cmap="viridis", origin="lower", extent=extent, vmin=0
    )
    axs[1, 1].set_title("Average 2D Power Spectrum (log)")
    axs[1, 1].set_xlabel("Frequency X")
    axs[1, 1].set_ylabel("Frequency Y")
    fig.colorbar(im, ax=axs[1, 1])
    set_integer_ticks(axs[1, 1])

    log_power_spectrum_sd = fft.fftshift(
        torch.log1p(torch.sqrt(torch.tensor(spectral.var_power_spectrum[channel])))
    )
    im = axs[1, 2].imshow(
        log_power_spectrum_sd,
        cmap="viridis",
        origin="lower",
        extent=extent,
        vmin=0,
    )
    axs[1, 2].set_title("2D Power Spectrum SD")
    axs[1, 2].set_xlabel("Frequency X")
    axs[1, 2].set_ylabel("Frequency Y")
    fig.colorbar(im, ax=axs[1, 2])
    set_integer_ticks(axs[1, 2])

    plt.tight_layout()
    return fig


def _plot_receptive_fields(ax: Axes, rf: FloatArray):
    """Plot full-color receptive field and individual color channels for CIFAR-10 range (-1 to 1)."""
    # Clear the main axes
    ax.clear()
    ax.axis("off")

    # Create a GridSpec within the given axes
    gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())

    rf_full = np.moveaxis(rf, 0, -1)  # Move channel axis to the last dimension
    rf_min = rf_full.min()
    rf_max = rf_full.max()
    rf_full = (rf_full - rf_min) / (rf_max - rf_min)
    # Full-color receptive field

    ax_full = ax.figure.add_subplot(gs[0, 0])
    ax_full.imshow(rf_full)
    ax_full.set_title("Full Color")
    ax_full.axis("off")

    # Individual color channels
    channels = ["Red", "Green", "Blue"]
    cmaps = ["RdGy_r", "RdYlGn", "PuOr"]  # Diverging colormaps centered at 0
    positions = [(0, 1), (1, 0), (1, 1)]  # Correct positions for a 2x2 grid
    for i in range(3):
        row, col = positions[i]
        ax_channel = ax.figure.add_subplot(gs[row, col])
        im = ax_channel.imshow(rf[i], cmap=cmaps[i], vmin=rf_min, vmax=rf_max)
        ax_channel.set_title(channels[i])
        ax_channel.axis("off")
        plt.colorbar(im, ax=ax_channel, fraction=0.046, pad=0.04)

    # Add min and max values as text
    ax.text(
        0.5,
        -0.05,
        f"Min: {rf.min():.2f}, Max: {rf.max():.2f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )


def analyze_input(
    device: torch.device, dataloader: DataLoader[tuple[Tensor, Tensor, int]]
) -> tuple[SpectralAnalysis, HistogramAnalysis]:
    spectral_result = _layer_spectral_analysis(device, dataloader, nn.Identity())
    histogram_result = _layer_pixel_histograms(device, dataloader, nn.Identity())
    return spectral_result, histogram_result


def input_plot(
    log: FigureLogger,
    rf_result: FloatArray,
    spectral_result: SpectralAnalysis,
    histogram_result: HistogramAnalysis,
    init_dir: str,
):
    for channel in range(histogram_result.channel_histograms.shape[0]):
        channel_fig = layer_channel_plots(
            rf_result,
            spectral_result,
            histogram_result,
            layer_name="input",
            channel=channel,
        )
        log.log_figure(
            channel_fig,
            init_dir,
            f"input_channel_{channel}",
            0,
            False,
        )
