import unittest
from unittest.mock import MagicMock, patch

import correct


class ProcessVideoTests(unittest.TestCase):
    def test_process_video_raises_when_video_writer_fails_to_open(self):
        mock_capture = MagicMock()
        mock_capture.get.side_effect = [1920, 1080]
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = False
        video_data = {
            "input_video_path": "input.mp4",
            "output_video_path": "output.mp4",
            "fps": 30,
            "frame_count": 1,
            "filters": [[1.0]],
            "filter_indices": [0],
        }

        with patch.object(correct.cv2, "VideoCapture", return_value=mock_capture), \
             patch.object(correct.cv2, "VideoWriter_fourcc", return_value=1234), \
             patch.object(correct.cv2, "VideoWriter", return_value=mock_writer):
            with self.assertRaisesRegex(
                RuntimeError,
                "Failed to open VideoWriter — check codec and output path",
            ):
                list(correct.process_video(video_data))

        mock_capture.release.assert_called_once()
        mock_writer.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
