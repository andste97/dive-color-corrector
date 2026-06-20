import unittest
from unittest.mock import patch

import numpy as np

import correct


class FakeCapture:
    def __init__(self, fps, frame_count):
        self.fps = fps
        self.frame_count = frame_count
        self.read_count = 0
        self.released = False

    def get(self, prop):
        if prop == correct.cv2.CAP_PROP_FPS:
            return self.fps
        if prop == correct.cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        return 0

    def isOpened(self):
        return self.read_count <= self.frame_count

    def read(self):
        if self.read_count < self.frame_count:
            self.read_count += 1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        self.read_count += 1
        return False, None

    def release(self):
        self.released = True


class AnalyzeVideoTests(unittest.TestCase):
    def test_preserves_fractional_fps_and_uses_integer_sampling_interval(self):
        capture = FakeCapture(fps=29.97, frame_count=60)
        filter_matrix = np.arange(20)

        with patch.object(correct.cv2, "VideoCapture", return_value=capture):
            with patch.object(correct, "get_filter_matrix", return_value=filter_matrix):
                with patch("builtins.print"):
                    final_result = list(
                        correct.analyze_video("in.mp4", "out.mp4")
                    )[-1]

        self.assertEqual(final_result["fps"], 29.97)
        self.assertEqual(final_result["filter_indices"], [60])
        np.testing.assert_array_equal(final_result["filters"], [filter_matrix])
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
