import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter


# Difference of Gaussians 2D function
def dog_2d(coords, amp1, amp2, x0, y0, sigma1_x, sigma1_y, sigma2_x, sigma2_y, offset, theta=0):
    """Difference of two 2D Gaussians"""
    x, y = coords
    x0 = float(x0)
    y0 = float(y0)
    
    # Rotation matrices for both Gaussians
    a1 = (np.cos(theta)**2)/(2*sigma1_x**2) + (np.sin(theta)**2)/(2*sigma1_y**2)
    b1 = -(np.sin(2*theta))/(4*sigma1_x**2) + (np.sin(2*theta))/(4*sigma1_y**2)
    c1 = (np.sin(theta)**2)/(2*sigma1_x**2) + (np.cos(theta)**2)/(2*sigma1_y**2)
    
    a2 = (np.cos(theta)**2)/(2*sigma2_x**2) + (np.sin(theta)**2)/(2*sigma2_y**2)
    b2 = -(np.sin(2*theta))/(4*sigma2_x**2) + (np.sin(2*theta))/(4*sigma2_y**2)
    c2 = (np.sin(theta)**2)/(2*sigma2_x**2) + (np.cos(theta)**2)/(2*sigma2_y**2)
    
    g1 = amp1 * np.exp(-(a1*(x-x0)**2 + 2*b1*(x-x0)*(y-y0) + c1*(y-y0)**2))
    g2 = amp2 * np.exp(-(a2*(x-x0)**2 + 2*b2*(x-x0)*(y-y0) + c2*(y-y0)**2))
    
    return offset + g1 - g2



def fit_dog_2d(image, blur_sigma=1):
    """Fit a Difference of Gaussians to image data"""
    
    img = gaussian_filter(np.abs(image), sigma=blur_sigma)
    ny, nx = img.shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    
    # Detect peak or dip
    peak_val = np.max(img)
    dip_val = np.min(img)
    mean_val = np.mean(img)
    
    if abs(peak_val - mean_val) > abs(dip_val - mean_val):
        peak_idx = np.unravel_index(np.argmax(img), img.shape)
        amp1_guess = (peak_val - mean_val) * 2  # Make center stronger
    else:
        peak_idx = np.unravel_index(np.argmin(img), img.shape)
        amp1_guess = (dip_val - mean_val) * 2
    
    peak_y, peak_x = peak_idx
    sigma_small = min(nx, ny) / 8  # Smaller center
    sigma_large = min(nx, ny) / 3  # Larger surround
    
    # Initial guess: [amp1, amp2, x0, y0, sig1_x, sig1_y, sig2_x, sig2_y, offset]
    initial_guess = [
        amp1_guess,           # amplitude of center Gaussian
        amp1_guess * 0.3,     # amplitude of surround (significantly smaller)
        peak_x,               # center x
        peak_y,               # center y
        sigma_small,          # center sigma x
        sigma_small,          # center sigma y
        sigma_large,          # surround sigma x (MUST be larger)
        sigma_large,          # surround sigma y
        mean_val              # offset
    ]
    
    # STRICT Bounds to enforce proper DoG structure
    lower_bounds = [
        -np.inf,              # amp1
        -np.inf,              # amp2
        0,                    # x0
        0,                    # y0
        0.5,                  # sigma1_x (minimum center width)
        0.5,                  # sigma1_y
        sigma_small * 1.5,    # sigma2_x MUST be at least 1.5x larger
        sigma_small * 1.5,    # sigma2_y MUST be at least 1.5x larger
        -np.inf               # offset
    ]
    
    upper_bounds = [
        np.inf,               # amp1
        np.inf,               # amp2
        nx,                   # x0
        ny,                   # y0
        sigma_large * 0.6,    # sigma1_x (center can't be too wide)
        sigma_large * 0.6,    # sigma1_y
        nx,                   # sigma2_x
        ny,                   # sigma2_y
        np.inf                # offset
    ]
    
    try:
        params, _ = optimize.curve_fit(
            lambda coords, *p: dog_2d(coords, *p, theta=0),
            (x.ravel(), y.ravel()),
            img.ravel(),
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )
    except Exception as e:
        print(f"DoG fit failed: {e}")
        print("Trying with relaxed constraints...")
        # Try again with slightly relaxed constraints
        lower_bounds[6] = sigma_small * 1.3  # Relax to 1.3x
        lower_bounds[7] = sigma_small * 1.3
        try:
            params, _ = optimize.curve_fit(
                lambda coords, *p: dog_2d(coords, *p, theta=0),
                (x.ravel(), y.ravel()),
                img.ravel(),
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=15000
            )
        except:
            print("Fit failed even with relaxed constraints, using initial guess")
            params = initial_guess
    
    param_dict = {
        'amp1': params[0],
        'amp2': params[1],
        'x0': params[2],
        'y0': params[3],
        'sigma1_x': params[4],
        'sigma1_y': params[5],
        'sigma2_x': params[6],
        'sigma2_y': params[7],
        'offset': params[8],
        'theta': 0
    }
    
    return param_dict


def rf_to_magnitude(rf_channel: FloatArray) -> np.ndarray:
    """Collapse multi-channel RF (C,H,W) to magnitude (H,W)."""
    return np.sqrt(np.sum(np.square(rf_channel), axis=0))


def _dog_map_from_params(shape: Tuple[int, int], params: Dict[str, float]) -> np.ndarray:
    """Generate DoG map from params on a given (H,W) grid."""
    h, w = shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return dog_2d(
        (x, y),
        params["amp1"],
        params["amp2"],
        params["x0"],
        params["y0"],
        params["sigma1_x"],
        params["sigma1_y"],
        params["sigma2_x"],
        params["sigma2_y"],
        params["offset"],
        params.get("theta", 0),
    )


def _r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute coefficient of determination."""
    sse = np.sum((target - prediction) ** 2)
    sst = np.sum((target - np.mean(target)) ** 2)
    if sst == 0:
        return 0.0
    return 1 - sse / sst


@dataclass
class DoGFitResult:
    params: Dict[str, float]
    fit_success: bool
    r2: float
    fitted_map: np.ndarray
    residual_map: np.ndarray


def fit_dog_to_channel(rf_channel: FloatArray, blur_sigma: float = 1) -> DoGFitResult:
    """Fit DoG to a single channel RF (C,H,W) using magnitude collapse."""
    mag = rf_to_magnitude(rf_channel)
    params = fit_dog_2d(mag, blur_sigma=blur_sigma)
    fitted = _dog_map_from_params(mag.shape, params)
    residual = mag - fitted
    r2 = _r2_score(mag, fitted)
    fit_success = np.isfinite(r2)
    return DoGFitResult(params, fit_success, float(r2), fitted, residual)


def analyze_layer(rf_layer: FloatArray, blur_sigma: float = 1) -> Dict[int, DoGFitResult]:
    """Fit DoG for every channel in a layer RF array (N,C,H,W)."""
    results: Dict[int, DoGFitResult] = {}
    for idx, rf_ch in enumerate(rf_layer):
        results[idx] = fit_dog_to_channel(rf_ch, blur_sigma=blur_sigma)
    return results


def analyze_all_layers(
    rf_result: Dict[str, FloatArray], blur_sigma: float = 1
) -> Dict[str, Dict[int, DoGFitResult]]:
    """Fit DoG for all layers."""
    return {layer: analyze_layer(rfs, blur_sigma=blur_sigma) for layer, rfs in rf_result.items()}


def _prepare_npz_dict(dog_results: Dict[str, Dict[int, DoGFitResult]]) -> Dict[str, Any]:
    """Flatten DoG results into NPZ-friendly dict of arrays."""
    npz_dict: Dict[str, Any] = {}
    for layer, layer_res in dog_results.items():
        ch_indices = sorted(layer_res.keys())

        def arr_from(key: str) -> np.ndarray:
            return np.array([layer_res[c].params[key] for c in ch_indices])

        npz_dict[f"{layer}_amp1"] = arr_from("amp1")
        npz_dict[f"{layer}_amp2"] = arr_from("amp2")
        npz_dict[f"{layer}_sigma1_x"] = arr_from("sigma1_x")
        npz_dict[f"{layer}_sigma1_y"] = arr_from("sigma1_y")
        npz_dict[f"{layer}_sigma2_x"] = arr_from("sigma2_x")
        npz_dict[f"{layer}_sigma2_y"] = arr_from("sigma2_y")
        npz_dict[f"{layer}_x0"] = arr_from("x0")
        npz_dict[f"{layer}_y0"] = arr_from("y0")
        npz_dict[f"{layer}_offset"] = arr_from("offset")
        npz_dict[f"{layer}_r2"] = np.array([layer_res[c].r2 for c in ch_indices])
        npz_dict[f"{layer}_fit_success"] = np.array(
            [layer_res[c].fit_success for c in ch_indices], dtype=bool
        )
        npz_dict[f"{layer}_residual_rms"] = np.array(
            [np.sqrt(np.mean(layer_res[c].residual_map**2)) for c in ch_indices]
        )
    return npz_dict


def _dog_stats_from_layer(layer_res: Dict[int, DoGFitResult]) -> Dict[str, float]:
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
    dog_results: Dict[str, Dict[int, DoGFitResult]],
    layer_name: str,
    max_cols: int = 8,
) -> Figure:
    """Plot overlays and residuals for channels in a layer."""
    rfs = rf_result[layer_name]
    layer_res = dog_results[layer_name]
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
        ax.set_title(f"Ch {idx} | R2={res.r2:.2f}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"DoG overlays & residuals — {layer_name}")
    fig.tight_layout()
    return fig


def plot_dog_statistics(dog_results: Dict[str, Dict[int, DoGFitResult]]) -> Figure:
    """Cross-layer summary of key metrics with distribution-aware visualizations."""
    layers = list(dog_results.keys())
    n_layers = len(layers)
    
    # Collect R² values per layer
    r2_per_layer = {l: np.array([r.r2 for r in dog_results[l].values()]) for l in layers}
    resid_per_layer = {
        l: np.array([np.sqrt(np.mean(r.residual_map**2)) for r in dog_results[l].values()])
        for l in layers
    }
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- Panel 1: R² distribution per layer (violin + swarm) ---
    ax1 = fig.add_subplot(gs[0, 0])

    # Prepare data as list of arrays for seaborn (no pandas needed)
    r2_arrays = [r2_per_layer[l] for l in layers]

    if any(len(arr) > 0 for arr in r2_arrays):
        positions = np.arange(n_layers)
        parts = ax1.violinplot(r2_arrays, positions=positions, showmeans=False, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        # Overlay individual points (swarm-like scatter with jitter)
        for i, (arr, pos) in enumerate(zip(r2_arrays, positions)):
            jitter = np.random.uniform(-0.15, 0.15, size=len(arr))
            ax1.scatter(pos + jitter, arr, color="black", s=10, alpha=0.5, zorder=3)
        ax1.set_xticks(positions)
        ax1.set_xticklabels(layers)

    ax1.set_ylim(-0.1, 1.05)
    ax1.axhline(0.7, color="red", linestyle="--", alpha=0.7, label="Good fit threshold (0.7)")
    ax1.set_title("R² Distribution by Layer")
    ax1.set_ylabel("R² (coefficient of determination)")
    ax1.tick_params(axis='x', rotation=20)
    ax1.legend(loc="lower right")
    
    # --- Panel 2: Fraction of channels with good DoG fit ---
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
    ax2.set_title("Fraction of Channels with Good DoG Fit")
    ax2.legend(loc="upper right")
    
    # --- Panel 3: R² histogram per layer (stacked/overlaid) ---
    ax3 = fig.add_subplot(gs[1, 0])
    
    bins = np.linspace(0, 1, 21)
    for l in layers:
        ax3.hist(r2_per_layer[l], bins=bins, alpha=0.5, label=l, density=True)
    
    ax3.set_xlabel("R²")
    ax3.set_ylabel("Density")
    ax3.set_title("R² Histogram by Layer")
    ax3.legend(loc="upper left")
    ax3.axvline(0.7, color="red", linestyle="--", alpha=0.7)
    
    # --- Panel 4: Summary statistics table ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    
    # Build summary table
    table_data = []
    headers = ["Layer", "n", "Median R²", "IQR", "% R²≥0.7", "Resid RMS"]
    
    for l in layers:
        r2_vals = r2_per_layer[l]
        resid_vals = resid_per_layer[l]
        q25, median, q75 = np.percentile(r2_vals, [25, 50, 75])
        pct_good = np.mean(r2_vals >= 0.7) * 100
        table_data.append([
            l,
            f"{len(r2_vals)}",
            f"{median:.3f}",
            f"[{q25:.2f}, {q75:.2f}]",
            f"{pct_good:.1f}%",
            f"{np.median(resid_vals):.3f}"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0"] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title("Summary Statistics (Median-based)", y=0.85)
    
    fig.suptitle("DoG Fitting Quality Analysis", fontsize=14, fontweight="bold")
    return fig

def plot_r2_history(r2_history: Dict[str, Any]) -> Figure:
    """Plot R² metrics over epochs per layer with distribution-aware tracking.
    
    r2_history format: {layer: [[epoch, r2_values_array], ...]} 
    where r2_values_array contains R² for all channels at that epoch.
    
    If using old format {layer: [[epoch, mean_r2], ...]}, falls back to simple plot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    
    # Check if we have full distribution data or just means
    has_full_data = False
    for layer, arr in r2_history.items():
        if len(arr) > 0 and len(arr[0]) > 1:
            if hasattr(arr[0][1], '__len__'):  # r2_values is array-like
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
        
        # Hide unused subplots
        for i in [(0, 1), (1, 0), (1, 1)]:
            axs[i].axis("off")
            axs[i].text(0.5, 0.5, "Upgrade to full R² tracking\nfor richer visualizations",
                       ha="center", va="center", transform=axs[i].transAxes)
        
        fig.suptitle("DoG R² History", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig
    
    # --- Panel 1: Median R² with IQR band ---
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
    ax1.axhline(0.7, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax1.legend(loc="lower right")
    set_integer_ticks(ax1)
    
    # --- Panel 2: Fraction of channels with R² >= threshold ---
    ax2 = axs[0, 1]
    threshold = 0.7
    
    for layer, history in r2_history.items():
        if len(history) == 0:
            continue
        epochs = [h[0] for h in history]
        fractions = [np.mean(np.array(h[1]) >= threshold) * 100 for h in history]
        ax2.plot(epochs, fractions, marker="s", label=layer, linewidth=2)
    
    ax2.set_title(f"% Channels with R² ≥ {threshold}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("% of channels")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right")
    set_integer_ticks(ax2)
    
    # --- Panel 3: Min/Max envelope showing range ---
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
    
    # --- Panel 4: Number of "good" vs "bad" channels stacked area ---
    ax4 = axs[1, 1]
    
    # Use first layer for demonstration, or combine all
    all_epochs = set()
    for history in r2_history.values():
        all_epochs.update(h[0] for h in history)
    all_epochs = sorted(all_epochs)
    
    if len(all_epochs) > 0:
        # Aggregate across layers
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
                     labels=["R² ≥ 0.7 (DoG-like)", "R² < 0.7 (non-DoG)"],
                     colors=["#66b3ff", "#ff9999"], alpha=0.8)
        ax4.set_title("Channel Classification Over Training")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Number of channels")
        ax4.legend(loc="center right")
        set_integer_ticks(ax4)
    
    fig.suptitle("DoG R² History Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

def _update_r2_history(
    r2_history: Dict[str, Any], dog_results: Dict[str, Dict[int, DoGFitResult]], epoch: int
) -> Dict[str, Any]:
    """Append full R² distribution for each layer to history."""
    updated = {k: [list(item) for item in v] for k, v in r2_history.items()}
    for layer, layer_res in dog_results.items():
        # Store full array of R² values, not just mean
        r2_values = np.array([r.r2 for r in layer_res.values()])
        updated.setdefault(layer, []).append([epoch, r2_values])
    return updated


def _load_r2_history(path) -> Dict[str, Any]:
    """Load R2 history if present."""
    try:
        dat = np.load(path, allow_pickle=True)
        return dat["r2_history"].item()
    except Exception:
        return {}


def _save_r2_history(path, r2_history: Dict[str, Any]) -> None:
    np.savez_compressed(path, r2_history=r2_history)


def plot(
    log: FigureLogger,
    rf_result: Dict[str, FloatArray],
    dog_results: Dict[str, Dict[int, DoGFitResult]],
    epoch: int,
    copy_checkpoint: bool,
    r2_history: Dict[str, Any],
):
    """Top-level plotting: overlays/residuals per layer, stats, and R2 history."""
    for layer in dog_results.keys():
        overlays = plot_layer_overlays(rf_result, dog_results, layer)
        log.log_figure(overlays, "dog", f"{layer}_overlays", epoch, copy_checkpoint)

    stats_fig = plot_dog_statistics(dog_results)
    log.log_figure(stats_fig, "dog", "dog_statistics", epoch, copy_checkpoint)

    r2_fig = plot_r2_history(r2_history)
    log.log_figure(r2_fig, "dog", "dog_r2_history", epoch, copy_checkpoint)


def to_npz_dict(dog_results: Dict[str, Dict[int, DoGFitResult]]) -> Dict[str, Any]:
    """Public helper to convert DoG results to savable dict."""
    return _prepare_npz_dict(dog_results)


def update_and_save_r2_history(
    history_path, dog_results: Dict[str, Dict[int, DoGFitResult]], epoch: int
) -> Dict[str, Any]:
    """Update R2 history on disk and return updated version."""
    existing = _load_r2_history(history_path)
    updated = _update_r2_history(existing, dog_results, epoch)
    _save_r2_history(history_path, updated)
    return updated

