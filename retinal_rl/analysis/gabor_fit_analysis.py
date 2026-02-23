"""Gabor fitting — thin wrapper around fit_analysis.py generic pipeline.

Provides only the Gabor-specific pieces:
  - fit_gabor_2d           (the actual curve_fit logic, uses math_utils.gabor_2d)
  - gabor_map_from_params (renders params back to a 2D image)

All shared infrastructure (FitResult, analyze_*, R² history, plotting)
lives in fit_analysis.py .
"""

from typing import Dict, Tuple

import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter

from retinal_rl.math_utils import estimate_frequency, estimate_orientation_fft, gabor_2d


# ---------------------------------------------------------------------------
# Gabor-specific fitting
# ---------------------------------------------------------------------------


def fit_gabor_2d(image: np.ndarray, blur_sigma: float = 0.5) -> Dict[str, float]:
    """Fit a 2D Gabor function to image data."""
    img = gaussian_filter(image, sigma=blur_sigma)
    ny, nx = img.shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    img_abs = np.abs(img - np.mean(img))
    total_mass = np.sum(img_abs)
    if total_mass > 0:
        y_cm = np.sum(y * img_abs) / total_mass
        x_cm = np.sum(x * img_abs) / total_mass
    else:
        y_cm, x_cm = ny / 2, nx / 2

    theta_guess = estimate_orientation_fft(img)
    freq_guess = estimate_frequency(img, theta_guess)
    sigma_guess = min(nx, ny) / 6
    amp_guess = np.max(np.abs(img - np.mean(img)))
    offset_guess = np.mean(img)

    initial_guess = [amp_guess, x_cm, y_cm, sigma_guess, sigma_guess,
                     freq_guess, theta_guess, 0.0, offset_guess]
    lower_bounds = [-np.inf, 0, 0, 0.5, 0.5, 0.001, -np.pi, -2 * np.pi, -np.inf]
    upper_bounds = [np.inf, nx, ny, nx, ny, 1.0, np.pi, 2 * np.pi, np.inf]

    try:
        params, _ = optimize.curve_fit(
            gabor_2d, (x.ravel(), y.ravel()), img.ravel(),
            p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=15000,
        )
    except Exception:
        try:
            initial_guess[7] = np.pi / 2
            params, _ = optimize.curve_fit(
                gabor_2d, (x.ravel(), y.ravel()), img.ravel(),
                p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=15000,
            )
        except Exception:
            params = initial_guess

    return {
        "amp": params[0], "x0": params[1], "y0": params[2],
        "sigma_x": params[3], "sigma_y": params[4],
        "freq": params[5], "theta": params[6],
        "phase": params[7], "offset": params[8],
    }


def gabor_map_from_params(shape: Tuple[int, int], params: Dict[str, float]) -> np.ndarray:
    """Render Gabor params to a 2D map on a (H, W) grid."""
    h, w = shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return gabor_2d(
        (x, y),
        params["amp"], params["x0"], params["y0"],
        params["sigma_x"], params["sigma_y"],
        params["freq"], params["theta"], params["phase"], params["offset"],
    )
