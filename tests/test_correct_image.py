"""End-to-end tests that exercise ``correct_image`` with the committed sample image."""

import os

import numpy as np
from PIL import Image

import correct


def test_correct_image_writes_output(sample_image_path, tmp_path):
    output_path = os.path.join(str(tmp_path), "corrected.png")
    preview_bytes = correct.correct_image(sample_image_path, output_path)

    assert os.path.isfile(output_path)
    assert isinstance(preview_bytes, (bytes, bytearray))
    assert len(preview_bytes) > 0

    with Image.open(sample_image_path) as original, Image.open(output_path) as corrected:
        assert corrected.size == original.size


def test_correct_image_boosts_average_red(sample_image_path, tmp_path):
    output_path = os.path.join(str(tmp_path), "corrected.png")
    correct.correct_image(sample_image_path, output_path)

    with Image.open(sample_image_path) as original:
        original_red = np.array(original.convert("RGB"))[..., 0].mean()
    with Image.open(output_path) as corrected:
        corrected_red = np.array(corrected.convert("RGB"))[..., 0].mean()

    assert corrected_red > original_red
