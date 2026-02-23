"""Reusable mathematical functions for receptive field analysis.

Contains model functions (DoG, Gabor), spectral estimation helpers,
and general-purpose metrics used across fit analysis modules.
"""

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# General-purpose utilities
# ---------------------------------------------------------------------------


def rf_to_magnitude(rf_channel: FloatArray) -> np.ndarray:
    """Collapse multi-channel RF (C, H, W) to magnitude (H, W)."""
    return np.sqrt(np.sum(np.square(rf_channel), axis=0))


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """Coefficient of determination (R²) between target and prediction."""
    sse = np.sum((target - prediction) ** 2)
    sst = np.sum((target - np.mean(target)) ** 2)
    if sst == 0:
        return 0.0
    return 1.0 - sse / sst


# ---------------------------------------------------------------------------
# 2D Gaussian (building block for DoG and standalone use)
# ---------------------------------------------------------------------------


def gaussian_2d(
    x: np.ndarray,
    y: np.ndarray,
    amp: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float = 0.0,
) -> np.ndarray:
    """2D Gaussian with optional rotation.

    Parameters
    ----------
    x, y : array
        Coordinate arrays.
    amp : float
        Amplitude.
    x0, y0 : float
        Center position.
    sigma_x, sigma_y : float
        Standard deviations along the two axes.
    theta : float
        Rotation angle (radians).
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    sin_2t = np.sin(2 * theta)

    a = cos_t**2 / (2 * sigma_x**2) + sin_t**2 / (2 * sigma_y**2)
    b = -sin_2t / (4 * sigma_x**2) + sin_2t / (4 * sigma_y**2)
    c = sin_t**2 / (2 * sigma_x**2) + cos_t**2 / (2 * sigma_y**2)

    return amp * np.exp(
        -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    )


# ---------------------------------------------------------------------------
# Difference of Gaussians (DoG) model
# ---------------------------------------------------------------------------


def dog_2d(
    coords,
    amp1: float,
    amp2: float,
    x0: float,
    y0: float,
    sigma1_x: float,
    sigma1_y: float,
    sigma2_x: float,
    sigma2_y: float,
    offset: float,
    theta: float = 0.0,
) -> np.ndarray:
    """2D Difference of Gaussians: center excitatory minus surround inhibitory.

    Parameters
    ----------
    coords : tuple of array
        ``(x, y)`` coordinate arrays.
    amp1, amp2 : float
        Amplitudes of center and surround Gaussians.
    x0, y0 : float
        Center position.
    sigma1_x, sigma1_y : float
        Std-devs of the center Gaussian.
    sigma2_x, sigma2_y : float
        Std-devs of the surround Gaussian.
    offset : float
        DC offset.
    theta : float
        Rotation angle (radians).
    """
    x, y = coords
    x0 = float(x0)
    y0 = float(y0)

    g1 = gaussian_2d(x, y, amp1, x0, y0, sigma1_x, sigma1_y, theta)
    g2 = gaussian_2d(x, y, amp2, x0, y0, sigma2_x, sigma2_y, theta)
    return offset + g1 - g2


# ---------------------------------------------------------------------------
# Gabor model
# ---------------------------------------------------------------------------


def gabor_2d(
    coords,
    amp: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    freq: float,
    theta: float,
    phase: float,
    offset: float,
) -> np.ndarray:
    """2D Gabor function: Gaussian-windowed sinusoidal grating.

    Parameters
    ----------
    coords : tuple of array
        ``(x, y)`` coordinate arrays.
    amp : float
        Amplitude.
    x0, y0 : float
        Center position.
    sigma_x, sigma_y : float
        Std-devs of the Gaussian envelope.
    freq : float
        Spatial frequency (cycles / pixel).
    theta : float
        Grating orientation (radians).
    phase : float
        Phase offset (radians).
    offset : float
        DC offset.
    """
    x, y = coords
    x0 = float(x0)
    y0 = float(y0)

    x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

    gaussian = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
    sinusoid = np.cos(2 * np.pi * freq * x_rot + phase)

    return offset + amp * gaussian * sinusoid


# ---------------------------------------------------------------------------
# Spectral estimation helpers (used by Gabor fitting)
# ---------------------------------------------------------------------------


def estimate_orientation_fft(image: np.ndarray) -> float:
    """Estimate dominant orientation of *image* via 2D FFT power spectrum.

    Returns orientation in radians.
    """
    f = np.fft.fft2(image - np.mean(image))
    fshift = np.fft.fftshift(f)
    power_spectrum = np.abs(fshift) ** 2

    ny, nx = image.shape
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    FY, FX = np.meshgrid(fy, fx, indexing="ij")

    # Zero out DC component
    cy, cx = ny // 2, nx // 2
    power_spectrum[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0

    angles = np.arctan2(FY, FX)
    total_power = np.sum(power_spectrum)

    if total_power > 0:
        sin_sum = np.sum(power_spectrum * np.sin(2 * angles))
        cos_sum = np.sum(power_spectrum * np.cos(2 * angles))
        return float(np.arctan2(sin_sum, cos_sum) / 2)

    return 0.0


def estimate_frequency(image: np.ndarray, theta: float) -> float:
    """Estimate spatial frequency along orientation *theta*.

    Parameters
    ----------
    image : 2-D array
        Receptive field image.
    theta : float
        Orientation in radians.

    Returns
    -------
    float
        Estimated frequency in cycles per pixel, clipped to [0.01, 0.5].
    """
    from scipy.ndimage import map_coordinates

    ny, nx = image.shape
    center_y, center_x = ny // 2, nx // 2

    max_dist = int(np.sqrt(nx**2 + ny**2) / 2)
    t = np.arange(-max_dist, max_dist)
    x_line = center_x + t * np.cos(theta)
    y_line = center_y + t * np.sin(theta)

    valid = (x_line >= 0) & (x_line < nx) & (y_line >= 0) & (y_line < ny)
    x_line = x_line[valid]
    y_line = y_line[valid]

    profile = map_coordinates(image, [y_line, x_line], order=1)
    profile = profile - np.mean(profile)

    fft_profile = np.fft.fft(profile)
    power = np.abs(fft_profile) ** 2
    freqs = np.fft.fftfreq(len(profile))

    power[0] = 0  # remove DC
    peak_idx = np.argmax(power[: len(power) // 2])

    freq = np.abs(freqs[peak_idx]) if peak_idx > 0 else 0.05
    return float(np.clip(freq, 0.01, 0.5))
