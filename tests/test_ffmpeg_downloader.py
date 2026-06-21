"""Tests for the FfmpegDownloader helper in ffmpeg_downloader.py."""

from ffmpeg_downloader import FfmpegDownloader


class _FakeResponse:
    def __init__(self, chunks, content_length):
        self._chunks = chunks
        self.headers = {"content-length": str(content_length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        return iter(self._chunks)


def test_download_reports_progress(tmp_path, monkeypatch):
    chunks = [b"a" * 100, b"b" * 50]
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(chunks, 150))

    progress = []
    downloader = FfmpegDownloader(lambda d, t: progress.append((d, t)))
    local_path = str(tmp_path / "ffmpeg.zip")
    returned = downloader._download("http://example/ffmpeg.zip", local_path)

    assert returned == local_path
    # Initial 0-byte call followed by one call per chunk.
    assert progress == [(0, 150), (100, 150), (150, 150)]
    with open(local_path, "rb") as f:
        assert f.read() == b"".join(chunks)


def test_ensure_available_without_callback_uses_library_directly(monkeypatch):
    import sys
    import types

    class _FakeRun:
        download_file = staticmethod(lambda url, path: path)

        @staticmethod
        def get_or_fetch_platform_executables_else_raise():
            return ("ffmpeg", "ffprobe")

    fake_static_ffmpeg = types.ModuleType("static_ffmpeg")
    fake_static_ffmpeg.run = _FakeRun
    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_static_ffmpeg)

    assert FfmpegDownloader().ensure_available() == ("ffmpeg", "ffprobe")
