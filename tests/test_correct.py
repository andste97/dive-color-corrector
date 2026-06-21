"""Tests for the video correction pipeline in correct.py.

These tests exercise the end-to-end video flow (analyze_video + process_video)
and the audio muxing helpers. The ffmpeg/ffprobe binaries are provided by the
bundled static-ffmpeg package, so no system-wide ffmpeg installation is needed.
"""

import os
import subprocess
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import correct


class AnalyzeVideoTests(unittest.TestCase):
    def test_preserves_fractional_fps_and_uses_integer_sampling_interval(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        capture = MagicMock()
        capture.get.side_effect = lambda prop: {
            correct.cv2.CAP_PROP_FPS: 29.97,
            correct.cv2.CAP_PROP_FRAME_COUNT: 60,
        }.get(prop, 0)
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, frame)] * 60 + [(False, None)]
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
        capture.release.assert_called_once()


def _make_source_video(path, ffmpeg, with_audio):
    """Create a small synthetic test video, optionally containing an audio track."""
    command = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
    ]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
        command += ["-c:a", "aac"]
    else:
        command += ["-an"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def _run_pipeline(input_path, output_path):
    """Run the full analyze + process pipeline for a single video."""
    video_data = None
    for item in correct.analyze_video(input_path, output_path):
        if isinstance(item, dict):
            video_data = item
    assert video_data is not None, "analyze_video did not yield video_data"
    # Consume the process_video generator to completion.
    list(correct.process_video(video_data))


@pytest.fixture(scope="module")
def ffmpeg_tools():
    """Resolve the bundled ffmpeg/ffprobe binaries, skipping if unavailable."""
    ffmpeg, ffprobe = correct.get_ffmpeg_executables()
    if not ffmpeg or not ffprobe:
        pytest.skip("bundled ffmpeg/ffprobe could not be obtained")
    return ffmpeg, ffprobe


def test_get_ffmpeg_executables_returns_existing_binaries(ffmpeg_tools):
    ffmpeg, ffprobe = ffmpeg_tools
    assert os.path.exists(ffmpeg)
    assert os.path.exists(ffprobe)


def test_has_audio_stream_detects_audio(ffmpeg_tools, tmp_path):
    ffmpeg, _ = ffmpeg_tools
    with_audio = str(tmp_path / "with_audio.mp4")
    without_audio = str(tmp_path / "without_audio.mp4")
    _make_source_video(with_audio, ffmpeg, with_audio=True)
    _make_source_video(without_audio, ffmpeg, with_audio=False)

    assert correct.has_audio_stream(with_audio) is True
    assert correct.has_audio_stream(without_audio) is False


def test_process_video_creates_output_with_audio(ffmpeg_tools, tmp_path):
    ffmpeg, _ = ffmpeg_tools
    source = str(tmp_path / "source_with_audio.mp4")
    output = str(tmp_path / "corrected_with_audio.mp4")
    _make_source_video(source, ffmpeg, with_audio=True)

    _run_pipeline(source, output)

    # The corrected output must exist and retain an audio track.
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0
    assert correct.has_audio_stream(output) is True

    # No temporary file should be left behind in the output directory.
    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".mp4")]
    assert sorted(leftovers) == ["corrected_with_audio.mp4", "source_with_audio.mp4"]


def test_process_video_creates_output_without_audio(ffmpeg_tools, tmp_path):
    ffmpeg, _ = ffmpeg_tools
    source = str(tmp_path / "source_no_audio.mp4")
    output = str(tmp_path / "corrected_no_audio.mp4")
    _make_source_video(source, ffmpeg, with_audio=False)

    _run_pipeline(source, output)

    # The output is still created, but without an audio track.
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0
    assert correct.has_audio_stream(output) is False

    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".mp4")]
    assert sorted(leftovers) == ["corrected_no_audio.mp4", "source_no_audio.mp4"]


def test_mux_audio_combines_video_and_audio(ffmpeg_tools, tmp_path):
    ffmpeg, _ = ffmpeg_tools
    source = str(tmp_path / "original.mp4")
    corrected = str(tmp_path / "corrected_video_only.mp4")
    output = str(tmp_path / "muxed.mp4")
    _make_source_video(source, ffmpeg, with_audio=True)
    # Build a video-only file to stand in for OpenCV's audio-less output.
    _make_source_video(corrected, ffmpeg, with_audio=False)

    assert correct.mux_audio(corrected, source, output) is True
    assert os.path.exists(output)
    assert correct.has_audio_stream(output) is True


def test_mux_audio_skips_when_source_has_no_audio(ffmpeg_tools, tmp_path):
    ffmpeg, _ = ffmpeg_tools
    source = str(tmp_path / "original_no_audio.mp4")
    corrected = str(tmp_path / "corrected_video_only.mp4")
    output = str(tmp_path / "muxed.mp4")
    _make_source_video(source, ffmpeg, with_audio=False)
    _make_source_video(corrected, ffmpeg, with_audio=False)

    # Positively audio-less source: muxing is skipped so the caller falls back.
    assert correct.mux_audio(corrected, source, output) is False


def test_mux_audio_returns_false_when_ffmpeg_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(correct, "get_ffmpeg_executables", lambda: (None, None))
    result = correct.mux_audio(
        str(tmp_path / "corrected.mp4"),
        str(tmp_path / "original.mp4"),
        str(tmp_path / "output.mp4"),
    )
    assert result is False


def test_ensure_ffmpeg_available_forwards_progress_callback(monkeypatch):
    chunks = [b"a" * 100, b"b" * 50]

    class _FakeResponse:
        headers = {"content-length": "150"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            return iter(chunks)

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse())

    class _FakeRun:
        download_file = staticmethod(lambda url, local_path: local_path)

        @staticmethod
        def get_or_fetch_platform_executables_else_raise():
            # Simulate the library invoking its (now patched) downloader.
            _FakeRun.download_file("http://example/ffmpeg.zip", os.devnull)
            return ("ffmpeg", "ffprobe")

    original_download = _FakeRun.download_file
    fake_static_ffmpeg = types.ModuleType("static_ffmpeg")
    fake_static_ffmpeg.run = _FakeRun
    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_static_ffmpeg)
    monkeypatch.setattr(correct, "_FFMPEG_EXECUTABLES", None)

    calls = []
    error = correct.ensure_ffmpeg_available(progress_callback=lambda d, t: calls.append((d, t)))

    assert error is None
    # The downloader streamed the file and reported byte progress.
    assert calls == [(0, 150), (100, 150), (150, 150)]
    # The original download_file must be restored after the call.
    assert _FakeRun.download_file is original_download


def test_ensure_ffmpeg_available_returns_error_message(monkeypatch):
    class _FailingRun:
        @staticmethod
        def get_or_fetch_platform_executables_else_raise():
            raise RuntimeError("network blocked")

    fake_static_ffmpeg = types.ModuleType("static_ffmpeg")
    fake_static_ffmpeg.run = _FailingRun
    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_static_ffmpeg)
    monkeypatch.setattr(correct, "_FFMPEG_EXECUTABLES", None)

    error = correct.ensure_ffmpeg_available()
    assert error == "network blocked"



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
            with self.assertRaisesRegex(RuntimeError, r"Failed to open VideoWriter"):
                list(correct.process_video(video_data))

        mock_capture.release.assert_called()
        mock_writer.release.assert_called()


if __name__ == "__main__":
    unittest.main()
