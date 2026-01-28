# This is the file for doing Gabor fits that start becoming interesting in V1

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import optimize
from scipy.ndimage import gaussian_filter

from retinal_rl.analysis.plot import FigureLogger, set_integer_ticks
from retinal_rl.util import FloatArray


def gabor_2d(
    coords,
    amp,
    x0,
    y0,
    sigma_x,
    sigma_y,
    freq,
    theta,
    phase,
    offset,
):
    """2D Gabor function: Gaussian-windowed sinusoidal grating.
    
    Args:
        coords: Tuple of (x, y) coordinate arrays
        amp: Amplitude of the Gabor
        x0, y0: Center position
        sigma_x, sigma_y: Standard deviations of Gaussian envelope
        freq: Spatial frequency of the sinusoid
        theta: Orientation of the grating (radians)
        phase: Phase offset of the sinusoid (radians)
        offset: DC offset
    """
    x, y = coords
    x0 = float(x0)
    y0 = float(y0)
    
    # Rotate coordinates to align with grating orientation
    x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    
    # Gaussian envelope
    gaussian = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
    
    # Sinusoidal grating
    sinusoid = np.cos(2 * np.pi * freq * x_rot + phase)
    
    return offset + amp * gaussian * sinusoid


def fit_gabor_2d(image: np.ndarray, blur_sigma: float = 0.5) -> Dict[str, float]:
    """Fit a 2D Gabor function to image data.
    
    Args:
        image: 2D array containing the receptive field
        blur_sigma: Gaussian blur to apply before fitting (reduces noise)
    
    Returns:
        Dictionary of fitted parameters
    """
    # Apply slight blur to reduce noise
    img = gaussian_filter(image, sigma=blur_sigma)
    ny, nx = img.shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    
    # Estimate center from center of mass
    img_abs = np.abs(img - np.mean(img))
    total_mass = np.sum(img_abs)
    if total_mass > 0:
        y_cm = np.sum(y * img_abs) / total_mass
        x_cm = np.sum(x * img_abs) / total_mass
    else:
        y_cm, x_cm = ny / 2, nx / 2
    
    # Estimate orientation using image moments or FFT
    theta_guess = _estimate_orientation_fft(img)
    
    # Estimate frequency from power spectrum
    freq_guess = _estimate_frequency(img, theta_guess)
    
    # Estimate spatial extent
    sigma_guess = min(nx, ny) / 6
    
    # Estimate amplitude and offset
    amp_guess = np.max(np.abs(img - np.mean(img)))
    offset_guess = np.mean(img)
    
    # Phase: try to detect if it's sine-phase (±π/2) or cosine-phase (0 or π)
    phase_guess = 0.0
    
    initial_guess = [
        amp_guess,      # amp
        x_cm,           # x0
        y_cm,           # y0
        sigma_guess,    # sigma_x
        sigma_guess,    # sigma_y
        freq_guess,     # freq
        theta_guess,    # theta
        phase_guess,    # phase
        offset_guess,   # offset
    ]
    
    # Set reasonable bounds
    lower_bounds = [
        -np.inf,        # amp (can be negative)
        0,              # x0
        0,              # y0
        0.5,            # sigma_x
        0.5,            # sigma_y
        0.001,          # freq (must be positive)
        -np.pi,         # theta
        -2*np.pi,       # phase
        -np.inf,        # offset
    ]
    
    upper_bounds = [
        np.inf,         # amp
        nx,             # x0
        ny,             # y0
        nx,             # sigma_x
        ny,             # sigma_y
        1.0,            # freq (cycles per pixel, should be < 0.5 by Nyquist)
        np.pi,          # theta
        2*np.pi,        # phase
        np.inf,         # offset
    ]
    
    try:
        params, _ = optimize.curve_fit(
            gabor_2d,
            (x.ravel(), y.ravel()),
            img.ravel(),
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=15000,
        )
    except Exception as e:
        # If fit fails, try with different initial conditions
        try:
            # Try different phase
            initial_guess[7] = np.pi / 2
            params, _ = optimize.curve_fit(
                gabor_2d,
                (x.ravel(), y.ravel()),
                img.ravel(),
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=15000,
            )
        except Exception:
            # If still fails, return initial guess
            params = initial_guess
    
    param_dict = {
        "amp": params[0],
        "x0": params[1],
        "y0": params[2],
        "sigma_x": params[3],
        "sigma_y": params[4],
        "freq": params[5],
        "theta": params[6],
        "phase": params[7],
        "offset": params[8],
    }
    
    return param_dict


def _estimate_orientation_fft(image: np.ndarray) -> float:
    """Estimate dominant orientation using 2D FFT power spectrum.
    
    Returns:
        Estimated orientation in radians
    """
    # Compute 2D FFT
    f = np.fft.fft2(image - np.mean(image))
    fshift = np.fft.fftshift(f)
    power_spectrum = np.abs(fshift) ** 2
    
    # Get coordinates in frequency domain
    ny, nx = image.shape
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    
    # Remove DC component
    center_y, center_x = ny // 2, nx // 2
    power_spectrum[center_y-2:center_y+3, center_x-2:center_x+3] = 0
    
    # Compute angle of each frequency component
    angles = np.arctan2(FY, FX)
    
    # Weight angles by power spectrum
    total_power = np.sum(power_spectrum)
    if total_power > 0:
        # Compute circular mean
        sin_sum = np.sum(power_spectrum * np.sin(2 * angles))
        cos_sum = np.sum(power_spectrum * np.cos(2 * angles))
        orientation = np.arctan2(sin_sum, cos_sum) / 2
    else:
        orientation = 0.0
    
    return orientation


def _estimate_frequency(image: np.ndarray, theta: float) -> float:
    """Estimate spatial frequency along the orientation theta.
    
    Args:
        image: 2D receptive field
        theta: Orientation in radians
    
    Returns:
        Estimated frequency in cycles per pixel
    """
    # Take 1D profile along orientation
    ny, nx = image.shape
    center_y, center_x = ny // 2, nx // 2
    
    # Create line coordinates along theta
    max_dist = int(np.sqrt(nx**2 + ny**2) / 2)
    t = np.arange(-max_dist, max_dist)
    x_line = center_x + t * np.cos(theta)
    y_line = center_y + t * np.sin(theta)
    
    # Sample image along line (with bounds checking)
    valid = (x_line >= 0) & (x_line < nx) & (y_line >= 0) & (y_line < ny)
    x_line = x_line[valid]
    y_line = y_line[valid]
    
    from scipy.ndimage import map_coordinates
    profile = map_coordinates(image, [y_line, x_line], order=1)
    
    # Remove mean
    profile = profile - np.mean(profile)
    
    # Compute FFT of profile
    fft_profile = np.fft.fft(profile)
    power = np.abs(fft_profile) ** 2
    freqs = np.fft.fftfreq(len(profile))
    
    # Find peak frequency (excluding DC)
    power[0] = 0  # Remove DC
    peak_idx = np.argmax(power[:len(power)//2])  # Only positive frequencies
    
    if peak_idx > 0:
        freq = np.abs(freqs[peak_idx])
    else:
        freq = 0.05  # Default fallback
    
    # Clip to reasonable range
    freq = np.clip(freq, 0.01, 0.5)
    
    return freq


def rf_to_magnitude(rf_channel: FloatArray) -> np.ndarray:
    """Collapse multi-channel RF (C,H,W) to magnitude (H,W)."""
    return np.sqrt(np.sum(np.square(rf_channel), axis=0))


def _gabor_map_from_params(shape: Tuple[int, int], params: Dict[str, float]) -> np.ndarray:
    """Generate Gabor map from params on a given (H,W) grid."""
    h, w = shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return gabor_2d(
        (x, y),
        params["amp"],
        params["x0"],
        params["y0"],
        params["sigma_x"],
        params["sigma_y"],
        params["freq"],
        params["theta"],
        params["phase"],
        params["offset"],
    )


def _r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute coefficient of determination."""
    sse = np.sum((target - prediction) ** 2)
    sst = np.sum((target - np.mean(target)) ** 2)
    if sst == 0:
        return 0.0
    return 1 - sse / sst


@dataclass
class GaborFitResult:
    """Container for Gabor fitting results."""
    params: Dict[str, float]
    fit_success: bool
    r2: float
    fitted_map: np.ndarray
    residual_map: np.ndarray


def fit_gabor_to_channel(rf_channel: FloatArray, blur_sigma: float = 0.5) -> GaborFitResult:
    """Fit Gabor to a single channel RF (C,H,W) using magnitude collapse.
    
    Args:
        rf_channel: Receptive field with shape (C, H, W)
        blur_sigma: Gaussian blur sigma for preprocessing
    
    Returns:
        GaborFitResult containing fitted parameters and quality metrics
    """
    mag = rf_to_magnitude(rf_channel)
    params = fit_gabor_2d(mag, blur_sigma=blur_sigma)
    fitted = _gabor_map_from_params(mag.shape, params)
    residual = mag - fitted
    r2 = _r2_score(mag, fitted)
    fit_success = np.isfinite(r2) and r2 > 0
    return GaborFitResult(params, fit_success, float(r2), fitted, residual)


def analyze_layer(rf_layer: FloatArray, blur_sigma: float = 0.5) -> Dict[int, GaborFitResult]:
    """Fit Gabor for every channel in a layer RF array (N,C,H,W).
    
    Args:
        rf_layer: Array of receptive fields with shape (N, C, H, W)
        blur_sigma: Gaussian blur sigma for preprocessing
    
    Returns:
        Dictionary mapping channel index to GaborFitResult
    """
    results: Dict[int, GaborFitResult] = {}
    for idx, rf_ch in enumerate(rf_layer):
        results[idx] = fit_gabor_to_channel(rf_ch, blur_sigma=blur_sigma)
    return results


def analyze_all_layers(
    rf_result: Dict[str, FloatArray], blur_sigma: float = 0.5
) -> Dict[str, Dict[int, GaborFitResult]]:
    """Fit Gabor for all layers.
    
    Args:
        rf_result: Dictionary mapping layer names to RF arrays
        blur_sigma: Gaussian blur sigma for preprocessing
    
    Returns:
        Nested dictionary: {layer_name: {channel_idx: GaborFitResult}}
    """
    return {layer: analyze_layer(rfs, blur_sigma=blur_sigma) for layer, rfs in rf_result.items()}


def _prepare_npz_dict(gabor_results: Dict[str, Dict[int, GaborFitResult]]) -> Dict[str, Any]:
    """Flatten Gabor results into NPZ-friendly dict of arrays."""
    npz_dict: Dict[str, Any] = {}
    for layer, layer_res in gabor_results.items():
        ch_indices = sorted(layer_res.keys())
        
        def arr_from(key: str) -> np.ndarray:
            return np.array([layer_res[c].params[key] for c in ch_indices])
        
        npz_dict[f"{layer}_amp"] = arr_from("amp")
        npz_dict[f"{layer}_x0"] = arr_from("x0")
        npz_dict[f"{layer}_y0"] = arr_from("y0")
        npz_dict[f"{layer}_sigma_x"] = arr_from("sigma_x")
        npz_dict[f"{layer}_sigma_y"] = arr_from("sigma_y")
        npz_dict[f"{layer}_freq"] = arr_from("freq")
        npz_dict[f"{layer}_theta"] = arr_from("theta")
        npz_dict[f"{layer}_phase"] = arr_from("phase")
        npz_dict[f"{layer}_offset"] = arr_from("offset")
        npz_dict[f"{layer}_r2"] = np.array([layer_res[c].r2 for c in ch_indices])
        npz_dict[f"{layer}_fit_success"] = np.array(
            [layer_res[c].fit_success for c in ch_indices], dtype=bool
        )
        npz_dict[f"{layer}_residual_rms"] = np.array(
            [np.sqrt(np.mean(layer_res[c].residual_map**2)) for c in ch_indices]
        )
    return npz_dict


def _gabor_stats_from_layer(layer_res: Dict[int, GaborFitResult]) -> Dict[str, float]:
    """Compute summary stats for a layer."""
    r2_vals = np.array([res.r2 for res in layer_res.values()])
    resid_rms = np.array([np.sqrt(np.mean(res.residual_map**2)) for res in layer_res.values()])
    return {
        "r2_mean": float(np.nanmean(r2_vals)),
        "r2_std": float(np.nanstd(r2_vals)),
        "resid_rms_mean": float(np.nanmean(resid_rms)),
        "resid_rms_std": float(np.nanstd(resid_rms)),
    }


def plot_layer_overlays(
    rf_result: Dict[str, FloatArray],
    gabor_results: Dict[str, Dict[int, GaborFitResult]],
    layer_name: str,
    max_cols: int = 8,
) -> Figure:
    """Plot overlays and residuals for channels in a layer.
    
    Shows the original RF magnitude with Gabor fit contours overlaid.
    """
    rfs = rf_result[layer_name]
    layer_res = gabor_results[layer_name]
    num_channels = len(layer_res)
    cols = min(num_channels, max_cols)
    rows = int(np.ceil(num_channels / cols))
    fig = plt.figure(figsize=(cols * 4, rows * 3))
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    
    for idx in range(num_channels):
        res = layer_res[idx]
        rf_mag = rf_to_magnitude(rfs[idx])
        ax = fig.add_subplot(gs[idx])
        
        # Show RF magnitude
        im = ax.imshow(rf_mag, cmap="RdBu_r", vmin=-np.max(np.abs(rf_mag)), 
                      vmax=np.max(np.abs(rf_mag)))
        
        # Overlay Gabor fit contours
        ax.contour(res.fitted_map, colors="lime", linewidths=0.8, levels=5)
        
        # Add orientation line
        params = res.params
        theta = params["theta"]
        theta_edge = theta + np.pi/2
        x0, y0 = params["x0"], params["y0"]
        length = 2 * max(params["sigma_x"], params["sigma_y"])
        dx = length * np.cos(theta_edge)
        dy = length * np.sin(theta_edge)
        ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], 
               'y-', linewidth=2, alpha=0.7)
        
        # Title with key parameters
        edge_angle = np.degrees(theta + np.pi/2) % 180  # Convert to edge orientation
        ax.set_title(
            f"Ch {idx} | R²={res.r2:.2f}\n"
            f"Edge θ={edge_angle:.0f}° f={params['freq']:.3f}",
            fontsize=9
        )
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    
    fig.suptitle(f"Gabor overlays — {layer_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_gabor_statistics(gabor_results: Dict[str, Dict[int, GaborFitResult]]) -> Figure:
    """Cross-layer summary of Gabor fit quality and parameters."""
    layers = list(gabor_results.keys())
    n_layers = len(layers)
    
    # Collect metrics per layer
    r2_per_layer = {l: np.array([r.r2 for r in gabor_results[l].values()]) for l in layers}
    freq_per_layer = {l: np.array([r.params["freq"] for r in gabor_results[l].values()]) for l in layers}
    theta_per_layer = {l: np.array([r.params["theta"] for r in gabor_results[l].values()]) for l in layers}
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- Panel 1: R² distribution per layer ---
    ax1 = fig.add_subplot(gs[0, 0])
    r2_arrays = [r2_per_layer[l] for l in layers]
    
    if any(len(arr) > 0 for arr in r2_arrays):
        positions = np.arange(n_layers)
        parts = ax1.violinplot(r2_arrays, positions=positions, showmeans=False, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        # Overlay individual points
        for i, (arr, pos) in enumerate(zip(r2_arrays, positions)):
            jitter = np.random.uniform(-0.15, 0.15, size=len(arr))
            ax1.scatter(pos + jitter, arr, color="black", s=10, alpha=0.5, zorder=3)
        ax1.set_xticks(positions)
        ax1.set_xticklabels(layers, rotation=20, ha="right")
    
    ax1.set_ylim(-0.1, 1.05)
    ax1.axhline(0.7, color="red", linestyle="--", alpha=0.7, label="Good fit (0.7)")
    ax1.set_title("R² Distribution by Layer")
    ax1.set_ylabel("R² (coefficient of determination)")
    ax1.legend(loc="lower right")
    
    # --- Panel 2: Fraction of Gabor-like channels ---
    ax2 = fig.add_subplot(gs[0, 1])
    thresholds = [0.5, 0.7, 0.9]
    colors = ["#ffcc99", "#66b3ff", "#99ff99"]
    width = 0.25
    x = np.arange(n_layers)
    
    for i, thresh in enumerate(thresholds):
        fractions = [np.mean(r2_per_layer[l] >= thresh) * 100 for l in layers]
        ax2.bar(x + i * width, fractions, width, label=f"R² ≥ {thresh}", color=colors[i])
    
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(layers, rotation=20, ha="right")
    ax2.set_ylabel("% of channels")
    ax2.set_ylim(0, 105)
    ax2.set_title("Fraction of Channels with Good Gabor Fit")
    ax2.legend(loc="upper right")
    
    # --- Panel 3: Spatial frequency distribution ---
    ax3 = fig.add_subplot(gs[0, 2])
    freq_arrays = [freq_per_layer[l] for l in layers]
    
    if any(len(arr) > 0 for arr in freq_arrays):
        positions = np.arange(n_layers)
        parts = ax3.violinplot(freq_arrays, positions=positions, showmeans=False, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        ax3.set_xticks(positions)
        ax3.set_xticklabels(layers, rotation=20, ha="right")
    
    ax3.set_title("Spatial Frequency Distribution")
    ax3.set_ylabel("Frequency (cycles/pixel)")
    ax3.set_ylim(0, 0.5)
    
    # --- Panel 4: Orientation distribution (circular histogram) ---
    ax4 = fig.add_subplot(gs[1, 0], projection='polar')
    
    # Combine all orientations across layers
    all_thetas = []
    all_weights = []  # Weight by R²
    for l in layers:
        thetas = theta_per_layer[l]
        r2s = r2_per_layer[l]
        all_thetas.extend(thetas)
        all_weights.extend(r2s)
    
    if len(all_thetas) > 0:
        bins = np.linspace(-np.pi, np.pi, 25)
        hist, bin_edges = np.histogram(all_thetas, bins=bins, weights=all_weights)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = bin_edges[1] - bin_edges[0]
        ax4.bar(bin_centers, hist, width=width, alpha=0.7, color='steelblue')
    
    ax4.set_title("Orientation Distribution (R²-weighted)", pad=20)
    ax4.set_theta_zero_location('E')
    ax4.set_theta_direction(1)
    
    # --- Panel 5: R² histogram ---
    ax5 = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 21)
    for l in layers:
        ax5.hist(r2_per_layer[l], bins=bins, alpha=0.5, label=l, density=True)
    
    ax5.set_xlabel("R²")
    ax5.set_ylabel("Density")
    ax5.set_title("R² Histogram by Layer")
    ax5.legend(loc="upper left", fontsize=8)
    ax5.axvline(0.7, color="red", linestyle="--", alpha=0.7)
    
    # --- Panel 6: Summary statistics table ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    
    table_data = []
    headers = ["Layer", "n", "Med R²", "% R²≥0.7", "Med Freq", "Resid RMS"]
    
    for l in layers:
        r2_vals = r2_per_layer[l]
        freq_vals = freq_per_layer[l]
        resid_vals = np.array([np.sqrt(np.mean(r.residual_map**2)) 
                               for r in gabor_results[l].values()])
        
        median_r2 = np.median(r2_vals)
        pct_good = np.mean(r2_vals >= 0.7) * 100
        median_freq = np.median(freq_vals)
        median_resid = np.median(resid_vals)
        
        table_data.append([
            l,
            f"{len(r2_vals)}",
            f"{median_r2:.3f}",
            f"{pct_good:.1f}%",
            f"{median_freq:.3f}",
            f"{median_resid:.3f}"
        ])
    
    table = ax6.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0"] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax6.set_title("Summary Statistics", y=0.85)
    
    fig.suptitle("Gabor Fitting Quality Analysis", fontsize=14, fontweight="bold")
    return fig


def plot_r2_history(r2_history: Dict[str, Any]) -> Figure:
    """Plot R² metrics over epochs per layer with distribution tracking."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    
    # Check if we have full distribution data
    has_full_data = False
    for layer, arr in r2_history.items():
        if len(arr) > 0 and len(arr[0]) > 1:
            if hasattr(arr[0][1], '__len__'):
                has_full_data = True
            break
    
    if not has_full_data:
        # Fallback: old format with just mean R²
        ax = axs[0, 0]
        for layer, arr in r2_history.items():
            if len(arr) == 0:
                continue
            np_arr = np.array(arr)
            ax.plot(np_arr[:, 0], np_arr[:, 1], marker="o", label=layer)
        ax.set_title("Mean R² over epochs (legacy format)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean R²")
        ax.set_ylim(0, 1)
        ax.legend()
        
        for i in [(0, 1), (1, 0), (1, 1)]:
            axs[i].axis("off")
            axs[i].text(0.5, 0.5, "Upgrade to full R² tracking\nfor richer visualizations",
                       ha="center", va="center", transform=axs[i].transAxes)
        
        fig.suptitle("Gabor R² History", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig
    
    # --- Panel 1: Median R² with IQR ---
    ax1 = axs[0, 0]
    for layer, history in r2_history.items():
        if len(history) == 0:
            continue
        epochs = [h[0] for h in history]
        medians = [np.median(h[1]) for h in history]
        q25 = [np.percentile(h[1], 25) for h in history]
        q75 = [np.percentile(h[1], 75) for h in history]
        
        ax1.plot(epochs, medians, marker="o", label=layer, linewidth=2)
        ax1.fill_between(epochs, q25, q75, alpha=0.2)
    
    ax1.set_title("Median R² with IQR")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("R²")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.7, color="red", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")
    set_integer_ticks(ax1)
    
    # --- Panel 2: Fraction of Gabor-like channels ---
    ax2 = axs[0, 1]
    threshold = 0.7
    
    for layer, history in r2_history.items():
        if len(history) == 0:
            continue
        epochs = [h[0] for h in history]
        fractions = [np.mean(np.array(h[1]) >= threshold) * 100 for h in history]
        ax2.plot(epochs, fractions, marker="s", label=layer, linewidth=2)
    
    ax2.set_title(f"% Channels with R² ≥ {threshold} (Gabor-like)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("% of channels")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right")
    set_integer_ticks(ax2)
    
    # --- Panel 3: Min/Max envelope ---
    ax3 = axs[1, 0]
    for layer, history in r2_history.items():
        if len(history) == 0:
            continue
        epochs = [h[0] for h in history]
        mins = [np.min(h[1]) for h in history]
        maxs = [np.max(h[1]) for h in history]
        medians = [np.median(h[1]) for h in history]
        
        color = ax3.plot(epochs, medians, marker="o", label=layer, linewidth=2)[0].get_color()
        ax3.fill_between(epochs, mins, maxs, alpha=0.15, color=color)
    
    ax3.set_title("R² Range (min-max envelope)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("R²")
    ax3.set_ylim(0, 1)
    ax3.legend(loc="lower right")
    set_integer_ticks(ax3)
    
    # --- Panel 4: Channel classification stacked area ---
    ax4 = axs[1, 1]
    
    all_epochs = set()
    for history in r2_history.values():
        all_epochs.update(h[0] for h in history)
    all_epochs = sorted(all_epochs)
    
    if len(all_epochs) > 0:
        good_counts = []
        bad_counts = []
        
        for epoch in all_epochs:
            good = 0
            bad = 0
            for layer, history in r2_history.items():
                for h in history:
                    if h[0] == epoch:
                        r2_vals = np.array(h[1])
                        good += np.sum(r2_vals >= 0.7)
                        bad += np.sum(r2_vals < 0.7)
                        break
            good_counts.append(good)
            bad_counts.append(bad)
        
        ax4.stackplot(all_epochs, good_counts, bad_counts,
                     labels=["R² ≥ 0.7 (Gabor-like)", "R² < 0.7 (non-Gabor)"],
                     colors=["#66b3ff", "#ff9999"], alpha=0.8)
        ax4.set_title("Channel Classification Over Training")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Number of channels")
        ax4.legend(loc="center right")
        set_integer_ticks(ax4)
    
    fig.suptitle("Gabor R² History Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _update_r2_history(
    r2_history: Dict[str, Any], gabor_results: Dict[str, Dict[int, GaborFitResult]], epoch: int
) -> Dict[str, Any]:
    """Append full R² distribution for each layer to history."""
    updated = {k: [list(item) for item in v] for k, v in r2_history.items()}
    for layer, layer_res in gabor_results.items():
        r2_values = np.array([r.r2 for r in layer_res.values()])
        updated.setdefault(layer, []).append([epoch, r2_values])
    return updated


def _load_r2_history(path) -> Dict[str, Any]:
    """Load R² history if present."""
    try:
        dat = np.load(path, allow_pickle=True)
        return dat["r2_history"].item()
    except Exception:
        return {}


def _save_r2_history(path, r2_history: Dict[str, Any]) -> None:
    """Save R² history to disk."""
    np.savez_compressed(path, r2_history=r2_history)


def plot(
    log: FigureLogger,
    rf_result: Dict[str, FloatArray],
    gabor_results: Dict[str, Dict[int, GaborFitResult]],
    epoch: int,
    copy_checkpoint: bool,
    r2_history: Dict[str, Any],
):
    """Top-level plotting: overlays/residuals per layer, stats, and R² history."""
    for layer in gabor_results.keys():
        overlays = plot_layer_overlays(rf_result, gabor_results, layer)
        log.log_figure(overlays, "gabor", f"{layer}_overlays", epoch, copy_checkpoint)
    
    stats_fig = plot_gabor_statistics(gabor_results)
    log.log_figure(stats_fig, "gabor", "gabor_statistics", epoch, copy_checkpoint)
    
    r2_fig = plot_r2_history(r2_history)
    log.log_figure(r2_fig, "gabor", "gabor_r2_history", epoch, copy_checkpoint)


def to_npz_dict(gabor_results: Dict[str, Dict[int, GaborFitResult]]) -> Dict[str, Any]:
    """Public helper to convert Gabor results to savable dict."""
    return _prepare_npz_dict(gabor_results)


def update_and_save_r2_history(
    history_path, gabor_results: Dict[str, Dict[int, GaborFitResult]], epoch: int
) -> Dict[str, Any]:
    """Update R² history on disk and return updated version."""
    existing = _load_r2_history(history_path)
    updated = _update_r2_history(existing, gabor_results, epoch)
    _save_r2_history(history_path, updated)
    return updated