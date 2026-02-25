"""Orbit image generation utilities for RCP VMS vibration data.

Convention (matches orbit_dataset_v2):
- DC offset removed before conversion
- Axis range : [-axis_lim, axis_lim] mils
- Grid size  : img_size × img_size pixels
- Pixel value: visit-count density, normalised to [0, 1] float32
"""

from __future__ import annotations

import numpy as np

# Default bearing XY channel pairs (0-indexed, even=X, odd=Y)
BEARING_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (4, 5), (10, 11), (16, 17))

DEFAULT_AXIS_LIM: float = 3.0   # mils
DEFAULT_IMG_SIZE: int = 256


def volt_to_mil(
    x: np.ndarray,
    y: np.ndarray,
    mils_per_v: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove DC offset and convert volt signals to mils displacement."""
    x_ac = x - x.mean()
    y_ac = y - y.mean()
    return x_ac * mils_per_v, y_ac * mils_per_v


def make_orbit_image(
    x_mil: np.ndarray,
    y_mil: np.ndarray,
    axis_lim: float = DEFAULT_AXIS_LIM,
    img_size: int = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """Generate a 2-D orbit density image from X-Y displacement signals.

    Parameters
    ----------
    x_mil, y_mil : displacement signals in mils (DC already removed)
    axis_lim     : half-width of the displayed orbit in mils
    img_size     : pixel dimension of the square output image

    Returns
    -------
    img : float32 ndarray, shape (img_size, img_size), values in [0, 1]
    """
    img = np.zeros((img_size, img_size), dtype=np.float32)

    scale = (img_size - 1) / (2.0 * axis_lim)
    px = ((x_mil + axis_lim) * scale).astype(np.int32)
    py = ((y_mil + axis_lim) * scale).astype(np.int32)

    mask = (px >= 0) & (px < img_size) & (py >= 0) & (py < img_size)
    np.add.at(img, (py[mask], px[mask]), 1.0)

    max_val = img.max()
    if max_val > 0.0:
        img /= max_val

    return img


def make_orbit_stack(
    channels: list[np.ndarray],
    pairs: tuple[tuple[int, int], ...] = BEARING_PAIRS,
    mils_per_v: float = 10.0,
    axis_lim: float = DEFAULT_AXIS_LIM,
    img_size: int = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """Build a (N_bearings, img_size, img_size) float32 orbit image stack.

    Parameters
    ----------
    channels   : list of per-channel signal arrays from
                 RCPVMSParser.read_all_channels()
    pairs      : sequence of (ch_x, ch_y) index tuples
    mils_per_v : sensitivity factor from BIN header (mils per volt)
    axis_lim   : half-width of the orbit axis in mils
    img_size   : pixel dimension of each orbit image

    Returns
    -------
    stack : ndarray, shape (len(pairs), img_size, img_size), float32
            Slices with missing or empty channels are left as zeros.
    """
    stack = np.zeros((len(pairs), img_size, img_size), dtype=np.float32)
    for i, (ch_x, ch_y) in enumerate(pairs):
        if ch_x >= len(channels) or ch_y >= len(channels):
            continue
        if len(channels[ch_x]) == 0 or len(channels[ch_y]) == 0:
            continue
        x_mil, y_mil = volt_to_mil(channels[ch_x], channels[ch_y], mils_per_v)
        stack[i] = make_orbit_image(x_mil, y_mil, axis_lim, img_size)
    return stack
