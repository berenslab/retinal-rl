"""DoG fitting — thin wrapper around fit_analysis.py generic pipeline.

Provides only the DoG-specific pieces:
  - fit_dog_2d            (the actual curve_fit logic, uses math_utils.dog_2d)
  - dog_map_from_params  (renders params back to a 2D image)

All shared infrastructure (FitResult, analyze_*, R² history, plotting)
lives in fit_analysis.py .
"""

from typing import Dict, Tuple

import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter

from retinal_rl.math_utils import dog_2d


# ---------------------------------------------------------------------------
# DoG-specific fitting
# ---------------------------------------------------------------------------


def fit_dog_2d(image: np.ndarray, blur_sigma: float = 1) -> Dict[str, float]:
    """Fit a Difference of Gaussians to image data."""
    img = gaussian_filter(np.abs(image), sigma=blur_sigma)
    ny, nx = img.shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    peak_val = np.max(img)
    dip_val = np.min(img)
    mean_val = np.mean(img)

    if abs(peak_val - mean_val) > abs(dip_val - mean_val):
        peak_idx = np.unravel_index(np.argmax(img), img.shape)
        amp1_guess = (peak_val - mean_val) * 2
    else:
        peak_idx = np.unravel_index(np.argmin(img), img.shape)
        amp1_guess = (dip_val - mean_val) * 2

    peak_y, peak_x = peak_idx
    sigma_small = min(nx, ny) / 8
    sigma_large = min(nx, ny) / 3

    initial_guess = [amp1_guess, amp1_guess * 0.3, peak_x, peak_y,
                     sigma_small, sigma_small, sigma_large, sigma_large, mean_val]
    lower_bounds = [-np.inf, -np.inf, 0, 0, 0.5, 0.5, sigma_small * 1.5, sigma_small * 1.5, -np.inf]
    upper_bounds = [np.inf, np.inf, nx, ny, sigma_large * 0.6, sigma_large * 0.6, nx, ny, np.inf]

    try:
        params, _ = optimize.curve_fit(
            lambda coords, *p: dog_2d(coords, *p, theta=0),
            (x.ravel(), y.ravel()), img.ravel(),
            p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=10000,
        )
    except Exception:
        lower_bounds[6] = sigma_small * 1.3
        lower_bounds[7] = sigma_small * 1.3
        try:
            params, _ = optimize.curve_fit(
                lambda coords, *p: dog_2d(coords, *p, theta=0),
                (x.ravel(), y.ravel()), img.ravel(),
                p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=15000,
            )
        except Exception:
            params = initial_guess

    return {
        "amp1": params[0], "amp2": params[1],
        "x0": params[2], "y0": params[3],
        "sigma1_x": params[4], "sigma1_y": params[5],
        "sigma2_x": params[6], "sigma2_y": params[7],
        "offset": params[8], "theta": 0.0,
    }


def dog_map_from_params(shape: Tuple[int, int], params: Dict[str, float]) -> np.ndarray:
    """Render DoG params to a 2D map on a (H, W) grid."""
    h, w = shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return dog_2d(
        (x, y),
        params["amp1"], params["amp2"],
        params["x0"], params["y0"],
        params["sigma1_x"], params["sigma1_y"],
        params["sigma2_x"], params["sigma2_y"],
        params["offset"], params.get("theta", 0),
    )
