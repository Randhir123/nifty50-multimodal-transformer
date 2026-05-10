"""Gramian Angular Field and Markov Transition Field image generation."""

import numpy as np
from scipy.ndimage import zoom
from pyts.image import GramianAngularField, MarkovTransitionField


def compute_gaf(series: np.ndarray, image_size: int = 32) -> np.ndarray:
    """Compute Gramian Angular Field (summation method) in [-1, 1].

    The series is normalized to [-1, 1] locally to prevent leakage.
    When image_size > len(series), computes at native resolution then zooms up.
    """
    min_val, max_val = series.min(), series.max()
    if max_val > min_val:
        series_norm = 2.0 * (series - min_val) / (max_val - min_val) - 1.0
    else:
        series_norm = np.zeros_like(series)

    effective = min(image_size, len(series))
    gaf = GramianAngularField(image_size=effective, method="summation")
    result = gaf.fit_transform(series_norm.reshape(1, -1))[0]
    if effective != image_size:
        result = zoom(result, image_size / effective)
    return np.clip(result, -1.0, 1.0)


def compute_mtf(series: np.ndarray, image_size: int = 32, n_bins: int = 8) -> np.ndarray:
    """Compute Markov Transition Field in [0, 1].

    When image_size > len(series), computes at native resolution then zooms up.
    """
    n_unique = len(np.unique(series))
    actual_bins = min(n_bins, max(2, n_unique))

    effective = min(image_size, len(series))
    mtf = MarkovTransitionField(image_size=effective, n_bins=actual_bins)
    try:
        result = mtf.fit_transform(series.reshape(1, -1))[0]
    except ValueError:
        return np.zeros((image_size, image_size), dtype=np.float64)
    if effective != image_size:
        result = zoom(result, image_size / effective)
    return np.clip(result, 0.0, 1.0)


def compute_stacked_image(ohlcv_window: np.ndarray, image_size: int = 32) -> np.ndarray:
    """Compute a 2-channel (GAF, MTF) image tensor from an OHLCV window.
    
    Currently extracts the 'close' price (index 3 as per standard OHLCV). 
    Future work: Use multiple input series (high, low, volume) as additional channels.
    
    Returns:
        np.ndarray: Shape (2, image_size, image_size)
    """
    if len(ohlcv_window) < image_size:
        # pyts interpolates natively, but logging a warning or raising is best practice.
        pass 
        
    close_series = ohlcv_window[:, 3]
    gaf = compute_gaf(close_series, image_size=image_size)
    mtf = compute_mtf(close_series, image_size=image_size)
    
    return np.stack([gaf, mtf], axis=0)