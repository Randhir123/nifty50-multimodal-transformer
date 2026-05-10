"""Tests for Gramian Angular Field and MTF computations."""

import numpy as np
from src.data.timeseries_images import compute_gaf, compute_mtf, compute_stacked_image


def test_gaf_shape_and_range():
    series = np.random.randn(20)
    gaf = compute_gaf(series, image_size=32)
    assert gaf.shape == (32, 32)
    assert np.min(gaf) >= -1.0
    assert np.max(gaf) <= 1.0


def test_mtf_shape_and_range():
    series = np.random.randn(20)
    mtf = compute_mtf(series, image_size=32, n_bins=8)
    assert mtf.shape == (32, 32)
    assert np.min(mtf) >= 0.0
    assert np.max(mtf) <= 1.0


def test_stacked_image_shape():
    ohlcv = np.random.randn(20, 5)
    stacked = compute_stacked_image(ohlcv, image_size=32)
    assert stacked.shape == (2, 32, 32)


def test_gaf_deterministic():
    series = np.random.randn(20)
    gaf1 = compute_gaf(series)
    gaf2 = compute_gaf(series)
    np.testing.assert_array_equal(gaf1, gaf2)


def test_negative_constant_series():
    series = np.ones(20)
    stacked = compute_stacked_image(np.column_stack([series]*5))
    assert not np.isnan(stacked).any()


def test_negative_no_global_stats_used():
    series1 = np.random.randn(20)
    series2 = np.random.randn(20) * 100 + 500
    
    gaf1_run1 = compute_gaf(series1)
    _ = compute_gaf(series2)
    gaf1_run2 = compute_gaf(series1)
    
    np.testing.assert_array_equal(gaf1_run1, gaf1_run2)