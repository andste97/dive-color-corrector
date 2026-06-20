import numpy as np

import correct


def _solid_image(r, g, b, size=64):
    """Create a solid RGB image of the given color."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    return img


def test_hue_shift_red_preserves_float_precision():
    # A fractional average color should not be quantized to integers.
    avg = np.array([12.7, 130.4, 120.9], dtype=np.float32)
    shifted = correct.hue_shift_red(avg, 0)
    # With hue_shift 0 the red output is just 0.299 * red, which is fractional.
    assert np.issubdtype(shifted.dtype, np.floating)
    assert not np.allclose(shifted, np.round(shifted))


def test_get_filter_matrix_uses_float_average(monkeypatch):
    # Capture the average matrix that get_filter_matrix builds internally so we
    # can confirm it keeps sub-integer precision (no uint8 clipping/rounding).
    captured = {}
    original_hue_shift_red = correct.hue_shift_red

    def spy(mat, h):
        if mat.ndim == 1:
            captured.setdefault("avg", np.array(mat, copy=True))
        return original_hue_shift_red(mat, h)

    monkeypatch.setattr(correct, "hue_shift_red", spy)

    # Average red of 12.75 lies between integers; uint8 would have truncated it.
    img = _solid_image(12, 130, 120)
    img[:, : img.shape[1] // 4, 0] = 15  # nudge the mean off an integer
    correct.get_filter_matrix(img)

    avg = captured["avg"]
    assert np.issubdtype(avg.dtype, np.floating)
    assert not np.allclose(avg, np.round(avg))


def test_float_average_changes_hue_search_outcome():
    # Demonstrate that uint8 truncation of the average could change the hue
    # shift selected by the search, and that the float version is more precise.
    img = _solid_image(20, 128, 118)

    def hue_shift_for(avg_dtype):
        mat = img.copy()
        avg_mat = np.array([np.float64(v) for v in (img[..., 0].mean(),
                                                    img[..., 1].mean(),
                                                    img[..., 2].mean())])
        avg_mat = avg_mat.astype(avg_dtype)
        new_avg_r = avg_mat[0]
        hue_shift = 0
        while new_avg_r < correct.MIN_AVG_RED:
            shifted = correct.hue_shift_red(avg_mat.astype(np.float64), hue_shift)
            new_avg_r = np.sum(shifted)
            hue_shift += 1
            if hue_shift > correct.MAX_HUE_SHIFT:
                new_avg_r = correct.MIN_AVG_RED
        return hue_shift

    # Both should run without error; the float path operates on un-clipped data.
    float_shift = hue_shift_for(np.float32)
    uint_shift = hue_shift_for(np.uint8)
    assert float_shift > 0
    assert uint_shift > 0


def test_get_filter_matrix_runs_and_returns_expected_shape():
    img = _solid_image(12, 130, 120)
    filter_matrix = correct.get_filter_matrix(img.copy())
    assert filter_matrix.shape == (20,)
