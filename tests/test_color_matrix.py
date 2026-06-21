"""Unit tests for the color matrix computation helpers in ``correct.py``."""

import math

import numpy as np

import correct


def test_hue_shift_red_shape_is_preserved():
    mat = np.full((4, 5, 3), 100, dtype=np.float32)
    shifted = correct.hue_shift_red(mat, 30)
    assert shifted.shape == mat.shape


def test_hue_shift_red_zero_shift_keeps_red_and_zeros_others():
    # With h == 0: U == 1, W == 0, so the red coefficient collapses to 1.0
    # and the green/blue coefficients collapse to 0.0.
    mat = np.dstack([
        np.full((3, 3), 40.0),   # Red
        np.full((3, 3), 80.0),   # Green
        np.full((3, 3), 200.0),  # Blue
    ])
    shifted = correct.hue_shift_red(mat, 0)
    np.testing.assert_allclose(shifted[..., 0], 40.0)
    np.testing.assert_allclose(shifted[..., 1], 0.0)
    np.testing.assert_allclose(shifted[..., 2], 0.0)


def test_hue_shift_red_matches_manual_formula():
    mat = np.dstack([
        np.full((2, 2), 50.0),
        np.full((2, 2), 60.0),
        np.full((2, 2), 70.0),
    ])
    h = 45
    u = math.cos(h * math.pi / 180)
    w = math.sin(h * math.pi / 180)
    expected_r = (0.299 + 0.701 * u + 0.168 * w) * 50.0
    expected_g = (0.587 - 0.587 * u + 0.330 * w) * 60.0
    expected_b = (0.114 - 0.114 * u - 0.497 * w) * 70.0

    shifted = correct.hue_shift_red(mat, h)
    np.testing.assert_allclose(shifted[..., 0], expected_r)
    np.testing.assert_allclose(shifted[..., 1], expected_g)
    np.testing.assert_allclose(shifted[..., 2], expected_b)


def test_apply_filter_returns_clipped_uint8():
    mat = np.full((6, 6, 3), 128, dtype=np.uint8)
    # An over-driven gain on every channel should saturate to 255 after clipping.
    filt = np.array([
        5, 0, 0, 0, 0,
        0, 5, 0, 0, 0,
        0, 0, 5, 0, 0,
        0, 0, 0, 1, 0,
    ], dtype=np.float32)
    out = correct.apply_filter(mat, filt)
    assert out.dtype == np.uint8
    assert out.shape == mat.shape
    assert out.min() >= 0 and out.max() <= 255
    assert np.all(out == 255)


def test_apply_filter_identity_like_matrix():
    mat = np.dstack([
        np.full((4, 4), 100, dtype=np.uint8),
        np.full((4, 4), 110, dtype=np.uint8),
        np.full((4, 4), 120, dtype=np.uint8),
    ])
    identity = np.array([
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
    ], dtype=np.float32)
    out = correct.apply_filter(mat, identity)
    np.testing.assert_array_equal(out, mat)


def test_get_filter_matrix_returns_20_finite_values(underwater_rgb):
    filt = correct.get_filter_matrix(underwater_rgb)
    assert filt.shape == (20,)
    assert np.all(np.isfinite(filt))


def test_correct_preserves_shape(underwater_rgb):
    corrected = correct.correct(underwater_rgb)
    assert corrected.shape == underwater_rgb.shape
    assert corrected.dtype == np.uint8


def test_correct_boosts_red_channel(underwater_rgb):
    # The whole point of the corrector is to recover the lost red channel,
    # so the corrected output should have a higher average red than the input.
    # `correct` returns a BGR image, so red is channel index 2.
    avg_red_before = float(underwater_rgb[..., 0].mean())
    corrected = correct.correct(underwater_rgb)
    avg_red_after = float(corrected[..., 2].mean())
    assert avg_red_after > avg_red_before
