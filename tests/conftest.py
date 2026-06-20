import os
import sys

import numpy as np
import pytest

# Make the project root importable so tests can `import correct`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_IMAGE_PATH = os.path.join(FIXTURES_DIR, "sample_underwater.png")


@pytest.fixture
def sample_image_path():
    """Path to a small committed sample image with a typical underwater color cast."""
    assert os.path.isfile(SAMPLE_IMAGE_PATH), "Sample fixture image is missing"
    return SAMPLE_IMAGE_PATH


@pytest.fixture
def underwater_rgb():
    """A small synthetic RGB image with a strong blue cast and deficient red channel."""
    rng = np.random.default_rng(0)
    height, width = 48, 64
    mat = np.zeros((height, width, 3), dtype=np.uint8)
    mat[..., 0] = np.clip(30 + rng.normal(0, 10, (height, width)), 0, 255)   # Red
    mat[..., 1] = np.clip(120 + rng.normal(0, 20, (height, width)), 0, 255)  # Green
    mat[..., 2] = np.clip(180 + rng.normal(0, 20, (height, width)), 0, 255)  # Blue
    return mat
