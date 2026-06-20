"""Unit tests for the per-frame filter-matrix interpolation in ``correct.py``.

``precompute_filter_matrices`` is the heart of the video pipeline: filter
matrices are only sampled every ``SAMPLE_SECONDS`` seconds and then linearly
interpolated to every frame.
"""

import numpy as np
import pytest

import correct


def _two_sample_matrices():
    first = np.zeros(20, dtype=float)
    second = np.arange(20, dtype=float)
    return np.array([first, second])


def test_precompute_shape_matches_frame_count():
    frame_count = 10
    filter_indices = [0, 9]
    matrices = _two_sample_matrices()
    out = correct.precompute_filter_matrices(frame_count, filter_indices, matrices)
    assert out.shape == (frame_count, 20)


def test_precompute_endpoints_match_samples():
    frame_count = 10
    filter_indices = [0, 9]
    matrices = _two_sample_matrices()
    out = correct.precompute_filter_matrices(frame_count, filter_indices, matrices)
    np.testing.assert_allclose(out[0], matrices[0])
    np.testing.assert_allclose(out[9], matrices[1])


def test_precompute_midpoint_is_linear_interpolation():
    frame_count = 11
    filter_indices = [0, 10]
    matrices = _two_sample_matrices()
    out = correct.precompute_filter_matrices(frame_count, filter_indices, matrices)
    # Halfway between frame 0 and frame 10 should be the average of the samples.
    np.testing.assert_allclose(out[5], (matrices[0] + matrices[1]) / 2)


def test_precompute_is_monotonic_for_increasing_samples():
    frame_count = 20
    filter_indices = [0, 19]
    matrices = _two_sample_matrices()
    out = correct.precompute_filter_matrices(frame_count, filter_indices, matrices)
    # Each column moves monotonically from the first sample to the second.
    diffs = np.diff(out, axis=0)
    assert np.all(diffs >= -1e-9)


def test_precompute_clamps_before_first_and_after_last_index():
    # np.interp holds the boundary values for frames outside the sampled range.
    frame_count = 12
    filter_indices = [3, 8]
    matrices = _two_sample_matrices()
    out = correct.precompute_filter_matrices(frame_count, filter_indices, matrices)
    np.testing.assert_allclose(out[0], matrices[0])   # before first sample
    np.testing.assert_allclose(out[11], matrices[1])  # after last sample


def test_precompute_three_samples_interpolation():
    frame_count = 21
    filter_indices = [0, 10, 20]
    matrices = np.array([
        np.zeros(20, dtype=float),
        np.full(20, 10.0),
        np.full(20, 5.0),
    ])
    out = correct.precompute_filter_matrices(frame_count, filter_indices, matrices)
    np.testing.assert_allclose(out[0], 0.0)
    np.testing.assert_allclose(out[10], 10.0)
    np.testing.assert_allclose(out[20], 5.0)
    np.testing.assert_allclose(out[5], 5.0)    # midway 0 -> 10
    np.testing.assert_allclose(out[15], 7.5)   # midway 10 -> 5
