"""Tests for the video correction pipeline in correct.py.

These tests exercise the end-to-end video flow (analyze_video + process_video)
and the audio muxing helpers. The ffmpeg/ffprobe binaries are provided by the
bundled static-ffmpeg package, so no system-wide ffmpeg installation is needed.
"""

import os
import subprocess

import pytest

import correct


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
