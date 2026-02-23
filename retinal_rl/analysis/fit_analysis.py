"""Generic infrastructure for 2D model fitting to receptive fields.

Provides the shared pipeline used by both DoG and Gabor fitting:
FitResult dataclass, per-channel / per-layer fitting, NPZ serialization,
R² history tracking, and common plotting routines.

Model-specific modules (dog_fit_analysis, gabor_fit_analysis) supply:
  - a ``fit_2d`` function  (image → param dict)
  - a ``map_from_params`` function  (shape, params → predicted map)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.figure import Figure

from retinal_rl.analysis.plot import FigureLogger, set_integer_ticks
from retinal_rl.math_utils import FloatArray, r2_score, rf_to_magnitude

# Type aliases for the two callables each model must provide.
Fit2DFn = Callable[[np.ndarray, float], Dict[str, float]]
MapFromParamsFn = Callable[[Tuple[int, int], Dict[str, float]], np.ndarray]


# ---------------------------------------------------------------------------
# FitResult — replaces DoGFitResult / GaborFitResult
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Container for a single-channel model fit."""

    params: Dict[str, float]
    fit_success: bool
    r2: float
    fitted_map: np.ndarray
    residual_map: np.ndarray


# ---------------------------------------------------------------------------
# Fitting pipeline
# ---------------------------------------------------------------------------


def fit_to_channel(
    rf_channel: FloatArray,
    fit_2d: Fit2DFn,
    map_from_params: MapFromParamsFn,
    blur_sigma: float = 0.5,
    r2_success_threshold: float = 0.0,
) -> FitResult:
    """Fit a model to a single RF channel (C, H, W) via magnitude collapse."""
    mag = rf_to_magnitude(rf_channel)
    params = fit_2d(mag, blur_sigma)
    fitted = map_from_params(mag.shape, params)
    residual = mag - fitted
    r2 = r2_score(mag, fitted)
    fit_success = np.isfinite(r2) and r2 > r2_success_threshold
    return FitResult(params, fit_success, float(r2), fitted, residual)


def analyze_layer(
    rf_layer: FloatArray,
    fit_2d: Fit2DFn,
    map_from_params: MapFromParamsFn,
    blur_sigma: float = 0.5,
) -> Dict[int, FitResult]:
    """Fit every channel in a layer RF array (N, C, H, W)."""
    return {
        idx: fit_to_channel(rf_ch, fit_2d, map_from_params, blur_sigma)
        for idx, rf_ch in enumerate(rf_layer)
    }


def analyze_all_layers(
    rf_result: Dict[str, FloatArray],
    fit_2d: Fit2DFn,
    map_from_params: MapFromParamsFn,
    blur_sigma: float = 0.5,
) -> Dict[str, Dict[int, FitResult]]:
    """Fit every channel in every layer."""
    return {
        layer: analyze_layer(rfs, fit_2d, map_from_params, blur_sigma)
        for layer, rfs in rf_result.items()
    }


# ---------------------------------------------------------------------------
# NPZ serialization
# ---------------------------------------------------------------------------


def prepare_npz_dict(
    results: Dict[str, Dict[int, FitResult]],
) -> Dict[str, Any]:
    """Flatten fit results into an NPZ-friendly dict of arrays."""
    npz_dict: Dict[str, Any] = {}
    for layer, layer_res in results.items():
        ch_indices = sorted(layer_res.keys())
        param_keys = list(layer_res[ch_indices[0]].params.keys())

        for key in param_keys:
            npz_dict[f"{layer}_{key}"] = np.array(
                [layer_res[c].params[key] for c in ch_indices]
            )

        npz_dict[f"{layer}_r2"] = np.array([layer_res[c].r2 for c in ch_indices])
        npz_dict[f"{layer}_fit_success"] = np.array(
            [layer_res[c].fit_success for c in ch_indices], dtype=bool
        )
        npz_dict[f"{layer}_residual_rms"] = np.array(
            [np.sqrt(np.mean(layer_res[c].residual_map ** 2)) for c in ch_indices]
        )
    return npz_dict


# ---------------------------------------------------------------------------
# R² history persistence
# ---------------------------------------------------------------------------


def load_r2_history(path) -> Dict[str, Any]:
    """Load R² history from disk (returns empty dict on failure)."""
    try:
        dat = np.load(path, allow_pickle=True)
        return dat["r2_history"].item()
    except Exception:
        return {}


def save_r2_history(path, r2_history: Dict[str, Any]) -> None:
    """Save R² history to disk."""
    np.savez_compressed(path, r2_history=r2_history)


def update_r2_history(
    r2_history: Dict[str, Any],
    results: Dict[str, Dict[int, FitResult]],
    epoch: int,
) -> Dict[str, Any]:
    """Append current epoch's R² values to the history."""
    updated = {k: [list(item) for item in v] for k, v in r2_history.items()}
    for layer, layer_res in results.items():
        r2_values = np.array([r.r2 for r in layer_res.values()])
        updated.setdefault(layer, []).append([epoch, r2_values])
    return updated


def update_and_save_r2_history(
    history_path,
    results: Dict[str, Dict[int, FitResult]],
    epoch: int,
) -> Dict[str, Any]:
    """Load, update, save, and return R² history."""
    existing = load_r2_history(history_path)
    updated = update_r2_history(existing, results, epoch)
    save_r2_history(history_path, updated)
    return updated


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def stats_from_layer(layer_res: Dict[int, FitResult]) -> Dict[str, float]:
    """Compute summary stats for a layer."""
    r2_vals = np.array([res.r2 for res in layer_res.values()])
    resid_rms = np.array(
        [np.sqrt(np.mean(res.residual_map ** 2)) for res in layer_res.values()]
    )
    return {
        "r2_mean": float(np.nanmean(r2_vals)),
        "r2_std": float(np.nanstd(r2_vals)),
        "resid_rms_mean": float(np.nanmean(resid_rms)),
        "resid_rms_std": float(np.nanstd(resid_rms)),
    }


# ---------------------------------------------------------------------------
# Plotting — shared across fit types
# ---------------------------------------------------------------------------


def plot_layer_overlays(
    rf_result: Dict[str, FloatArray],
    fit_results: Dict[str, Dict[int, FitResult]],
    layer_name: str,
    fit_name: str,
    max_cols: int = 8,
) -> Figure:
    """Plot RF magnitude with fit contours overlaid for each channel."""
    rfs = rf_result[layer_name]
    layer_res = fit_results[layer_name]
    num_channels = len(layer_res)
    cols = min(num_channels, max_cols)
    rows = int(np.ceil(num_channels / cols))
    fig = plt.figure(figsize=(cols * 4, rows * 3))
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    for idx in range(num_channels):
        res = layer_res[idx]
        rf_mag = rf_to_magnitude(rfs[idx])
        ax = fig.add_subplot(gs[idx])
        im = ax.imshow(rf_mag, cmap="viridis")
        ax.contour(res.fitted_map, colors="white", linewidths=0.6)
        ax.set_title(f"Ch {idx} | R²={res.r2:.2f}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"{fit_name} overlays — {layer_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_r2_statistics(
    fit_results: Dict[str, Dict[int, FitResult]],
    fit_name: str,
) -> Figure:
    """Cross-layer R² violin plot."""
    layers = list(fit_results.keys())
    n_layers = len(layers)

    r2_per_layer = {
        l: np.array([r.r2 for r in fit_results[l].values()]) for l in layers
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    r2_arrays = [r2_per_layer[l] for l in layers]
    if any(len(arr) > 0 for arr in r2_arrays):
        positions = np.arange(n_layers)
        parts = ax.violinplot(
            r2_arrays, positions=positions, showmeans=False, showmedians=True
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        for arr, pos in zip(r2_arrays, positions):
            jitter = np.random.uniform(-0.15, 0.15, size=len(arr))
            ax.scatter(pos + jitter, arr, color="black", s=10, alpha=0.5, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(layers, rotation=20, ha="right")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0.7, color="red", linestyle="--", alpha=0.7, label="Good fit (0.7)")
    ax.set_title(f"{fit_name} R² Distribution by Layer")
    ax.set_ylabel("R² (coefficient of determination)")
    ax.legend(loc="lower right")

    fig.tight_layout()
    return fig


def plot_r2_history(r2_history: Dict[str, Any], fit_name: str) -> Figure:
    """Plot median R² with IQR over epochs."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Detect whether we have full distribution data or legacy means
    has_full_data = False
    for layer, arr in r2_history.items():
        if len(arr) > 0 and len(arr[0]) > 1:
            if hasattr(arr[0][1], "__len__"):
                has_full_data = True
        break

    if not has_full_data:
        for layer, arr in r2_history.items():
            if len(arr) == 0:
                continue
            np_arr = np.array(arr)
            ax.plot(np_arr[:, 0], np_arr[:, 1], marker="o", label=layer)
        ax.set_title(f"{fit_name} Mean R² over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean R²")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        return fig

    for layer, history in r2_history.items():
        if len(history) == 0:
            continue
        epochs = [h[0] for h in history]
        medians = [np.median(h[1]) for h in history]
        q25 = [np.percentile(h[1], 25) for h in history]
        q75 = [np.percentile(h[1], 75) for h in history]
        ax.plot(epochs, medians, marker="o", label=layer, linewidth=2)
        ax.fill_between(epochs, q25, q75, alpha=0.2)
    ax.set_title(f"{fit_name} Median R² with IQR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1)
    ax.axhline(0.7, color="red", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    set_integer_ticks(ax)

    fig.tight_layout()
    return fig


def plot(
    log: FigureLogger,
    rf_result: Dict[str, FloatArray],
    fit_results: Dict[str, Dict[int, FitResult]],
    epoch: int,
    copy_checkpoint: bool,
    r2_history: Dict[str, Any],
    fit_name: str,
):
    """Top-level plotting: overlays per layer, R² statistics, and R² history."""
    tag = fit_name.lower()

    for layer in fit_results.keys():
        overlays = plot_layer_overlays(rf_result, fit_results, layer, fit_name)
        log.log_figure(overlays, tag, f"{layer}_overlays", epoch, copy_checkpoint)

    stats_fig = plot_r2_statistics(fit_results, fit_name)
    log.log_figure(stats_fig, tag, f"{tag}_statistics", epoch, copy_checkpoint)

    r2_fig = plot_r2_history(r2_history, fit_name)
    log.log_figure(r2_fig, tag, f"{tag}_r2_history", epoch, copy_checkpoint)




