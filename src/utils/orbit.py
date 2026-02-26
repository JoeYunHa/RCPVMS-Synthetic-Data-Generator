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
    adaptive: bool = True,
    adaptive_percentile: float = 99.5,
    adaptive_min: float = 0.1,
) -> np.ndarray:
    """Build a (N_bearings, img_size, img_size) float32 orbit image stack.

    Parameters
    ----------
    channels             : list of per-channel signal arrays
    pairs                : sequence of (ch_x, ch_y) index tuples
    mils_per_v           : sensitivity factor from BIN header (mils per volt)
    axis_lim             : half-width in mils — used as-is when adaptive=False,
                           or as fallback when no valid channels exist
    img_size             : pixel dimension of each orbit image
    adaptive             : if True (default), compute a per-file global axis_lim
                           from the p99.5 orbit radius across ALL valid bearing
                           pairs, then apply the same scale to every channel.
                           Preserves inter-bearing relative amplitude while
                           preventing clipping / empty-image artefacts.
    adaptive_percentile  : percentile of orbit radius used for auto-scaling
    adaptive_min         : minimum axis_lim to avoid extreme zoom on noise (mils)

    Returns
    -------
    stack : ndarray, shape (len(pairs), img_size, img_size), float32
            Slices with missing or empty channels are left as zeros.
    """
    # ── Step 1: convert all valid pairs to mils (single pass) ────────────────
    mils_data: list[tuple[int, np.ndarray, np.ndarray]] = []
    for i, (ch_x, ch_y) in enumerate(pairs):
        if ch_x >= len(channels) or ch_y >= len(channels):
            continue
        if len(channels[ch_x]) == 0 or len(channels[ch_y]) == 0:
            continue
        x_mil, y_mil = volt_to_mil(channels[ch_x], channels[ch_y], mils_per_v)
        mils_data.append((i, x_mil, y_mil))

    # ── Step 2: compute per-file global axis_lim (Option A adaptive) ─────────
    if adaptive and mils_data:
        all_radii = np.concatenate([
            np.sqrt(x ** 2 + y ** 2) for _, x, y in mils_data
        ])
        computed_lim = float(np.percentile(all_radii, adaptive_percentile))
        axis_lim = max(computed_lim, adaptive_min)

    # ── Step 3: render all channels with the shared axis_lim ─────────────────
    stack = np.zeros((len(pairs), img_size, img_size), dtype=np.float32)
    for i, x_mil, y_mil in mils_data:
        stack[i] = make_orbit_image(x_mil, y_mil, axis_lim, img_size)
    return stack
