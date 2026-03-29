import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from retinal_rl.classification.imageset import Imageset, ImageSubset
from retinal_rl.analysis.plot import FigureLogger, set_integer_ticks
from retinal_rl.models.brain import Brain, get_cnn_circuit
from retinal_rl.util import is_nonlinearity

logger = logging.getLogger(__name__)

#----Inter-channel analysis: decorrelation analysis for convolutional layers----

def prepare_dataset(
    imageset: Imageset, max_sample_size: int = 0, batch_size: int = 64, num_workers: int = 0
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

    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def compute_decorrelation_scores(
    device: torch.device,
    dataloader: DataLoader[tuple[Tensor, Tensor, int]],
    brain: Brain,
) -> dict[str, tuple[float, float]]:
    """Compute channel decorrelation scores for each convolutional layer.

    Decorrelation score = 1 - mean(|off-diagonal Pearson correlations|)
    Uncertainty based on channel equalization (CV of per-channel standard deviations)
    Both range from 0 to 1.

    Args:
        device: torch device to run on
        dataloader: DataLoader yielding (observations, labels, indices)
        brain: The neural network model

    Returns:
        dict mapping layer_name -> (decorrelation_score, uncertainty) tuple
    """
    brain.eval()
    brain.to(device)

    _, cnn_layers = get_cnn_circuit(brain)
    results: dict[str, tuple[float, float]] = {}
    head_layers: list[nn.Module] = []

    for layer_name, layer in cnn_layers.items():
        head_layers.append(layer)
        if is_nonlinearity(layer):
            continue
        model = nn.Sequential(*head_layers)
        results[layer_name] = _layer_decorrelation_score(device, dataloader, model)

    return results


def _layer_decorrelation_score(
    device: torch.device,
    dataloader: DataLoader[tuple[Tensor, Tensor, int]],
    model: nn.Module,
    eps: float = 1e-8,
) -> float:
    """Compute decorrelation score for a single layer via online covariance accumulation.

    Accumulates cross-channel covariance matrix in O(C^2) memory instead of O(N*C).

    Args:
        device: torch device
        dataloader: DataLoader yielding (observations, labels, indices)
        model: Forward model ending at the target layer
        eps: Numerical stability constant for correlation computation

    Returns:
        Decorrelation score (float in [0, 1])
    """
    model.eval()
    model.to(device)

    sum_x = None  # shape (C,)
    sum_x2 = None  # shape (C, C) - sum of outer products
    n = 0

    with torch.no_grad():
        # First pass: probe first batch to get number of channels
        for batch in dataloader:
            observations = batch[0].to(device)
            activations = model(observations)  # shape (B, C, H, W)

            B, C, H, W = activations.shape

            # If single channel, no decorrelation to measure
            if C == 1:
                return 1.0

            # Flatten spatial dimensions: (B, C, H, W) -> (B*H*W, C)
            activations_flat = activations.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

            # Initialize accumulators on first batch
            if sum_x is None:
                sum_x = torch.zeros(C, device=device, dtype=activations.dtype)
                sum_x2 = torch.zeros((C, C), device=device, dtype=activations.dtype)

            # Accumulate
            sum_x += activations_flat.sum(dim=0)  # (C,)
            sum_x2 += activations_flat.T @ activations_flat  # (C, C)
            n += activations_flat.shape[0]

    # Compute covariance from sums
    mean = sum_x / n  # (C,)
    cov = sum_x2 / n - torch.outer(mean, mean)  # (C, C)

    # Convert covariance to correlation
    std = torch.sqrt(torch.diagonal(cov).clamp(min=eps))  # (C,) - clamp to avoid division by zero
    corr = cov / torch.outer(std, std)  # (C, C)

    # Extract off-diagonal absolute correlations
    mask = ~torch.eye(C, device=device, dtype=torch.bool)
    off_diag_corr = corr[mask]
    mean_abs_off_diag = torch.abs(off_diag_corr).mean().item()

    # Decorrelation score: 1 - mean(|off-diagonal correlations|)
    decorr_score = 1.0 - mean_abs_off_diag

    cv = std.std() / std.mean()
    equalization_score = 1 / (1 + cv)
    uncertainty = 1.0 - equalization_score

    return decorr_score, uncertainty


def update_and_save_decorrelation_history(
    path: Path,
    scores: dict[str, tuple[float, float]],
    epoch: int,
) -> dict[str, list]:
    """Update and save decorrelation history across epochs.

    Loads existing history from NPZ file, appends current epoch's scores and uncertainties,
    saves back to file, and returns the full history.

    Args:
        path: Path to decorrelation_history.npz file
        scores: dict mapping layer_name -> (decorrelation_score, uncertainty) tuple
        epoch: Current epoch number

    Returns:
        Full history dict: layer_name -> list of [epoch, score, uncertainty] triples
    """
    # Load existing history or start fresh
    if path.exists():
        data = dict(np.load(path, allow_pickle=True))
        history = data["decorrelation_history"].item()
    else:
        history = {}

    # Append current epoch scores and uncertainties
    for layer_name, (score, uncertainty) in scores.items():
        history.setdefault(layer_name, []).append([epoch, float(score), float(uncertainty)])

    # Save updated history
    np.savez_compressed(path, decorrelation_history=history)

    return history


#----Intra-channel analysis: spectral slope----

def compute_spectral_slopes(
    device: torch.device,
    dataloader: DataLoader[tuple[Tensor, Tensor, int]],
    brain: Brain,
) -> dict[str, tuple[float, float]]:
    """Compute power spectral slope per layer.

    Fits a line to the radially-averaged power spectrum in log-log frequency space.
    Slope ~0 = whitened; slope ~-2 = natural images (1/f^2 power law).

    Args:
        device: torch device to run on
        dataloader: DataLoader yielding (observations, labels, indices)
        brain: The neural network model

    Returns:
        dict mapping layer_name -> (mean_slope, std_slope) across channels
    """
    brain.eval()
    brain.to(device)

    _, cnn_layers = get_cnn_circuit(brain)
    results: dict[str, tuple[float, float]] = {}
    head_layers: list[nn.Module] = []

    for layer_name, layer in cnn_layers.items():
        head_layers.append(layer)
        if is_nonlinearity(layer):
            continue
        model = nn.Sequential(*head_layers)
        results[layer_name] = _layer_spectral_slope(device, dataloader, model)

    return results


def _layer_spectral_slope(
    device: torch.device,
    dataloader: DataLoader[tuple[Tensor, Tensor, int]],
    model: nn.Module,
    num_bins: int = 20,
    eps: float = 1e-8,
) -> tuple[float, float]:
    """Compute mean and std of log-log spectral slope across channels for one layer."""
    model.eval()
    model.to(device)

    sum_power = None   # (C, num_bins)
    bin_idx = None     # precomputed freq -> bin mapping
    bin_counts = None  # number of freq pixels per bin
    log_bin_centers = None
    n_batches = 0
    C = None

    with torch.no_grad():
        for batch in dataloader:
            obs = batch[0].to(device)
            acts = model(obs)  # (B, C, H, W)
            B, C_curr, H, W = acts.shape

            # Compute 2D FFT power spectrum
            fft = torch.fft.rfft2(acts)               # (B, C, H, W//2+1)
            power = fft.real ** 2 + fft.imag ** 2     # (B, C, H, W//2+1)

            # Build radial frequency grid and bin mapping once on first batch
            if sum_power is None:
                C = C_curr
                fy = torch.fft.fftfreq(H, device=device)
                fx = torch.fft.rfftfreq(W, device=device)
                fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")  # (H, W//2+1)
                r = torch.sqrt(fy_grid ** 2 + fx_grid ** 2).reshape(-1)   # (H*(W//2+1),)

                # Log-spaced bin edges from min nonzero freq to max
                r_min = r[r > 0].min().item()
                r_max = r.max().item()
                bin_edges = torch.logspace(
                    np.log10(r_min), np.log10(r_max), num_bins + 1, device=device
                )
                log_bin_centers = (
                    0.5 * (torch.log10(bin_edges[:-1]) + torch.log10(bin_edges[1:]))
                ).cpu().numpy()  # (num_bins,)

                # Precompute bin index for each freq pixel (DC at r=0 maps to bin 0)
                bin_idx = torch.bucketize(r, bin_edges[1:]).clamp(0, num_bins - 1)  # (N_freq,)
                bin_counts = torch.bincount(bin_idx, minlength=num_bins).float().clamp(min=1)

                sum_power = torch.zeros(C, num_bins, device=device)

            # Vectorized radial averaging: (B, C, N_freq) -> (C, num_bins)
            N_freq = H * (W // 2 + 1)
            power_flat = power.reshape(B, C, N_freq)  # (B, C, N_freq)

            bin_idx_bc = bin_idx.unsqueeze(0).unsqueeze(0).expand(B, C, -1)  # (B, C, N_freq)
            bin_power = torch.zeros(B, C, num_bins, device=device)
            bin_power.scatter_add_(2, bin_idx_bc, power_flat)
            bin_power = bin_power / bin_counts  # normalize by pixels per bin

            sum_power += bin_power.mean(dim=0)  # average over batch, accumulate
            n_batches += 1

    # Mean power across all batches, fit log-log slope per channel
    mean_power = (sum_power / n_batches).cpu().numpy()  # (C, num_bins)
    log_power = np.log10(np.maximum(mean_power, eps))   # (C, num_bins)

    slopes = np.array([
        np.polyfit(log_bin_centers, log_power[c], 1)[0]
        for c in range(C)
    ])

    return float(slopes.mean()), float(slopes.std())


def update_and_save_spectral_slope_history(
    path: Path,
    slopes: dict[str, tuple[float, float]],
    epoch: int,
) -> dict[str, list]:
    """Update and save spectral slope history across epochs.

    Args:
        path: Path to spectral_slope_history.npz file
        slopes: dict mapping layer_name -> (mean_slope, std_slope)
        epoch: Current epoch number

    Returns:
        Full history dict: layer_name -> list of [epoch, mean_slope, std_slope] triples
    """
    if path.exists():
        data = dict(np.load(path, allow_pickle=True))
        history = data["spectral_slope_history"].item()
    else:
        history = {}

    for layer_name, (mean_slope, std_slope) in slopes.items():
        history.setdefault(layer_name, []).append([epoch, float(mean_slope), float(std_slope)])

    np.savez_compressed(path, spectral_slope_history=history)

    return history


def plot(
    log: FigureLogger,
    decorr_history: dict[str, list],
    spectral_history: dict[str, list],
    epoch: int,
    copy_checkpoint: bool,
) -> None:
    """Plot both decorrelation and spectral slope analyses side-by-side.

    Creates a 1x2 figure with decorrelation scores (left) and spectral slopes (right).

    Args:
        log: FigureLogger instance for logging plots
        decorr_history: dict mapping layer_name -> list of [epoch, score, uncertainty] triples
        spectral_history: dict mapping layer_name -> list of [epoch, mean_slope, std_slope] triples
        epoch: Current epoch (for logging)
        copy_checkpoint: Whether to copy plot to checkpoint directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Left: Decorrelation =====
    for layer_name, layer_history in decorr_history.items():
        if len(layer_history) == 0:
            continue
        arr = np.array(layer_history)
        epochs = arr[:, 0]
        scores = arr[:, 1]
        uncertainties = arr[:, 2]

        # Normalize uncertainty for alpha scaling
        u_norm = (uncertainties - uncertainties.min()) / (uncertainties.ptp() + 1e-8)
        alpha_vals = 0.4 + 0.6 * (1 - u_norm)

        # Plot segment-wise to vary alpha
        for i in range(len(epochs) - 1):
            ax1.plot(
                epochs[i:i+2],
                scores[i:i+2],
                color=f"C{list(decorr_history.keys()).index(layer_name)}",
                alpha=alpha_vals[i],
                linewidth=2
            )

        # markers
        ax1.scatter(epochs, scores, s=20, alpha=0.9)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Decorrelation Score")
    ax1.set_title("Channel Decorrelation Score")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="lower right", frameon=False)
    ax1.grid(True, alpha=0.2)
    set_integer_ticks(ax1)

    #===== Right: Spectral Slope =====

    for i, (layer_name, layer_history) in enumerate(spectral_history.items()):
        if len(layer_history) == 0:
            continue

        arr = np.array(layer_history)
        epochs = arr[:, 0]
        slopes = arr[:, 1]

        ax2.plot(
            epochs,
            slopes,
            label=layer_name,
            linewidth=2,
            alpha=0.85,
            marker="o",
            markersize=3
        )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Spectral Slope")
    ax2.set_title("Power Spectral Slope")

    # cleaner legend
    ax2.legend(loc="lower right", frameon=False)

    # subtle grid + remove top/right borders
    ax2.grid(True, alpha=0.2)
    ax2.spines[['top', 'right']].set_visible(False)

    set_integer_ticks(ax2)

    fig.tight_layout()
    log.log_figure(fig, "channel_analysis", "combined_analysis", epoch, copy_checkpoint)