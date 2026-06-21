"""Unit tests for the histogram normalizing-interval logic in ``correct.py``."""

import numpy as np

import correct


def test_normalizing_interval_finds_largest_gap():
    # Gaps: 10, 1, 39 -> the largest gap is between 11 and 50.
    array = np.array([0, 10, 11, 50], dtype=float)
    low, high = correct.normalizing_interval(array)
    assert (low, high) == (11, 50)


def test_normalizing_interval_first_gap_when_largest_is_first():
    array = np.array([0, 100, 110, 120], dtype=float)
    low, high = correct.normalizing_interval(array)
    assert (low, high) == (0, 100)


def test_normalizing_interval_constant_array_returns_defaults():
    # No positive gap exists, so the function falls back to its defaults.
    array = np.full(256, 5.0)
    low, high = correct.normalizing_interval(array)
    assert (low, high) == (0, 255)


def test_normalizing_interval_low_is_strictly_below_high():
    rng = np.random.default_rng(7)
    array = np.sort(rng.integers(0, 256, size=64).astype(float))
    low, high = correct.normalizing_interval(array)
    assert high > low
