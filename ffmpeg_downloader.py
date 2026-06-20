"""Locate (and download when needed) the bundled static-ffmpeg binaries.

static-ffmpeg fetches the platform ffmpeg/ffprobe binaries on first use but
offers no progress hook. ``FfmpegDownloader`` wraps it and can report download
progress so a GUI can show how far along the download is.
"""

DOWNLOAD_CHUNK_SIZE = 256 * 1024  # 256 KB, matching static-ffmpeg's own chunking
DOWNLOAD_TIMEOUT = 10 * 60  # seconds


class FfmpegDownloader:
    """Resolve the bundled ffmpeg/ffprobe binaries, optionally reporting progress.

    ``progress_callback`` is invoked as ``progress_callback(downloaded_bytes,
    total_bytes)`` while a download is in progress (``total_bytes`` is 0 when the
    server does not report a content length). It is not called when the binaries
    are already cached locally.
    """

    def __init__(self, progress_callback=None):
        self._progress_callback = progress_callback

    def ensure_available(self):
        """Return the ``(ffmpeg, ffprobe)`` executable paths, downloading if needed.

        Raises if the binaries cannot be obtained.
        """
        from static_ffmpeg import run as static_ffmpeg_run

        if self._progress_callback is None:
            return static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()

        # static-ffmpeg has no progress hook, so temporarily swap in our
        # progress-aware downloader. Extraction, caching, locking and
        # permissions are still handled by the library.
        original_download = static_ffmpeg_run.download_file
        static_ffmpeg_run.download_file = self._download
        try:
            return static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
        finally:
            static_ffmpeg_run.download_file = original_download

    def _download(self, url, local_path):
        """Stream ``url`` to ``local_path``, reporting progress per chunk."""
        import requests

        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as req:
            req.raise_for_status()
            try:
                total = int(req.headers.get("content-length", 0))
            except (TypeError, ValueError):
                total = 0
            downloaded = 0
            self._progress_callback(downloaded, total)
            with open(local_path, "wb") as file_d:
                for chunk in req.iter_content(DOWNLOAD_CHUNK_SIZE):
                    file_d.write(chunk)
                    downloaded += len(chunk)
                    self._progress_callback(downloaded, total)
        return local_path
