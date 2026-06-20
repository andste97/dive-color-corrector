import unittest
from unittest.mock import patch

import numpy as np

import correct


class FakeCapture:
    def __init__(self, frames, *, fps=30, reported_frame_count=None):
        self.frames = frames
        self.fps = fps
        self.reported_frame_count = reported_frame_count or len(frames)
        self.read_count = 0
        self.released = False

    def get(self, prop):
        if prop == correct.cv2.CAP_PROP_FPS:
            return self.fps
        if prop == correct.cv2.CAP_PROP_FRAME_COUNT:
            return self.reported_frame_count
        if prop == correct.cv2.CAP_PROP_FRAME_WIDTH:
            return self.frames[0].shape[1] if self.frames else 0
        if prop == correct.cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0] if self.frames else 0
        return 0

    def isOpened(self):
        return self.read_count <= len(self.frames)

    def read(self):
        if self.read_count < len(self.frames):
            frame = self.frames[self.read_count]
            self.read_count += 1
            return True, frame
        self.read_count += 1
        return False, None

    def release(self):
        self.released = True


class FakeVideoWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class VideoCorrectionTests(unittest.TestCase):
    def test_analyze_video_samples_last_frame_for_short_video(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        capture = FakeCapture([frame], fps=30, reported_frame_count=1)
        filter_matrix = np.arange(20)

        with patch.object(correct.cv2, "VideoCapture", return_value=capture):
            with patch.object(correct, "get_filter_matrix", return_value=filter_matrix):
                with patch("builtins.print"):
                    video_data = list(
                        correct.analyze_video("short.mp4", "out.mp4")
                    )[-1]

        self.assertEqual(video_data["filter_indices"], [1])
        np.testing.assert_array_equal(video_data["filters"], [filter_matrix])
        self.assertTrue(capture.released)

    def test_precompute_filter_matrices_broadcasts_single_sample(self):
        filter_matrix = np.arange(20)

        interpolated = correct.precompute_filter_matrices(
            frame_count=3,
            filter_indices=[1],
            filter_matrices=np.array([filter_matrix]),
        )

        self.assertEqual(interpolated.shape, (3, 20))
        np.testing.assert_array_equal(interpolated[0], filter_matrix)
        np.testing.assert_array_equal(interpolated[1], filter_matrix)
        np.testing.assert_array_equal(interpolated[2], filter_matrix)

    def test_process_video_reuses_last_matrix_for_extra_frames(self):
        frames = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
        ]
        capture = FakeCapture(frames, fps=30, reported_frame_count=1)
        writer = FakeVideoWriter()
        video_data = {
            "input_video_path": "short.mp4",
            "output_video_path": "out.mp4",
            "fps": 30,
            "frame_count": 1,
            "filters": np.array([np.arange(20)]),
            "filter_indices": [1],
        }

        with patch.object(correct.cv2, "VideoCapture", return_value=capture):
            with patch.object(correct.cv2, "VideoWriter", return_value=writer):
                with patch.object(correct.cv2, "VideoWriter_fourcc", return_value=0):
                    with patch.object(
                        correct, "apply_filter", side_effect=lambda mat, _: mat
                    ):
                        with patch("builtins.print"):
                            results = list(correct.process_video(video_data))

        self.assertEqual(results, [None, None])
        self.assertEqual(len(writer.frames), 2)
        self.assertTrue(capture.released)
        self.assertTrue(writer.released)


if __name__ == "__main__":
    unittest.main()
