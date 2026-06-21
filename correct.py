import sys
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import math
from PIL import Image

THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2
SAMPLE_SECONDS = 2 # Extracts color correction from every N seconds

def hue_shift_red(mat, h):

    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)

    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]

    return np.dstack([r, g, b])

def normalizing_interval(array):

    high = 255
    low = 0
    max_dist = 0

    for i in range(1, len(array)):
        dist = array[i] - array[i-1]
        if(dist > max_dist):
            max_dist = dist
            high = array[i]
            low = array[i-1]

    return (low, high)

def apply_filter(mat, filt):
    filtered_mat = np.zeros_like(mat, dtype=np.float32)
    filtered_mat[..., 0] = mat[..., 0] * filt[0] + mat[..., 1] * filt[1] + mat[..., 2] * filt[2] + filt[4] * 255
    filtered_mat[..., 1] = mat[..., 1] * filt[6] + filt[9] * 255
    filtered_mat[..., 2] = mat[..., 2] * filt[12] + filt[14] * 255
    return np.clip(filtered_mat, 0, 255).astype(np.uint8)

def get_filter_matrix(mat):

    mat = cv2.resize(mat, (256, 256))

    # Get average values of RGB
    avg_mat = np.array(cv2.mean(mat)[:3], dtype=np.float32)
    
    # Find hue shift so that average red reaches MIN_AVG_RED
    new_avg_r = avg_mat[0]
    hue_shift = 0
    while(new_avg_r < MIN_AVG_RED):

        shifted = hue_shift_red(avg_mat, hue_shift)
        new_avg_r = np.sum(shifted)
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            new_avg_r = MIN_AVG_RED

    # Apply hue shift to whole image and replace red channel
    shifted_mat = hue_shift_red(mat, hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    new_r_channel = np.clip(new_r_channel, 0, 255)
    mat[..., 0] = new_r_channel

    # Get histogram of all channels
    hist_r = hist = cv2.calcHist([mat], [0], None, [256], [0,256])
    hist_g = hist = cv2.calcHist([mat], [1], None, [256], [0,256])
    hist_b = hist = cv2.calcHist([mat], [2], None, [256], [0,256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat.shape[0]*mat.shape[1])/THRESHOLD_RATIO
    for x in range(256):
        
        if hist_r[x] < threshold_level:
            normalize_mat[x][0] = x

        if hist_g[x] < threshold_level:
            normalize_mat[x][1] = x

        if hist_b[x] < threshold_level:
            normalize_mat[x][2] = x

    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])


    shifted = hue_shift_red(np.array([1, 1, 1]), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    red_gain = 256 / (adjust_r_high - adjust_r_low)
    green_gain = 256 / (adjust_g_high - adjust_g_low)
    blue_gain = 256 / (adjust_b_high - adjust_b_low)

    redOffset = (-adjust_r_low / 256) * red_gain
    greenOffset = (-adjust_g_low / 256) * green_gain
    blueOffset = (-adjust_b_low / 256) * blue_gain

    adjust_red = shifted_r * red_gain
    adjust_red_green = shifted_g * red_gain
    adjust_red_blue = shifted_b * red_gain * BLUE_MAGIC_VALUE

    return np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain, 0, 0, greenOffset,
        0, 0, blue_gain, 0, blueOffset,
        0, 0, 0, 1, 0,
    ])

def correct(mat):
    original_mat = mat.copy()

    filter_matrix = get_filter_matrix(mat)
    
    corrected_mat = apply_filter(original_mat, filter_matrix)
    corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)

    return corrected_mat

def correct_image(input_path, output_path):
    exif_data = None
    with Image.open(input_path) as image:
        exif_data = image.info.get("exif")
        if image.mode != "RGB":
            image = image.convert("RGB")
        mat = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    corrected_mat = correct(rgb_mat)

    output_image = Image.fromarray(cv2.cvtColor(corrected_mat, cv2.COLOR_BGR2RGB))
    save_kwargs = {}
    if exif_data:
        save_kwargs["exif"] = exif_data
    output_image.save(output_path, **save_kwargs)
    
    preview = mat.copy()
    width = preview.shape[1] // 2
    preview[::, width:] = corrected_mat[::, width:]

    preview = cv2.resize(preview, (960, 540))

    return cv2.imencode('.png', preview)[1].tobytes()


def analyze_video(input_video_path, output_video_path):
    
    # Initialize new video writer
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = max(1, round(fps * SAMPLE_SECONDS)) if fps > 0 else 1
    frame_count = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get filter matrices for every 10th frame
    filter_matrix_indexes = []
    filter_matrices = []
    count = 0
    
    print("Analyzing...")
    while(cap.isOpened()):
        
        count += 1  
        print(f"{count} frames", end="\r")
        ret, frame = cap.read()
        if not ret:
            # End video read if we have gone beyond reported frame count
            if count >= frame_count:
                break

            # Failsafe to prevent an infinite loop
            if count >= 1e6:
                break

            # Otherwise this is just a faulty frame read, try reading next frame
            continue

        # Pick filter matrix from every N seconds
        if count % sample_interval == 0:
            mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            filter_matrix_indexes.append(count) 
            filter_matrices.append(get_filter_matrix(mat))

        yield count
        
    cap.release()

    # Build a interpolation function to get filter matrix at any given frame
    filter_matrices = np.array(filter_matrices)

    yield {
        "input_video_path": input_video_path,
        "output_video_path": output_video_path,
        "fps": fps,
        "frame_count": count,
        "filters": filter_matrices,
        "filter_indices": filter_matrix_indexes
    }

def precompute_filter_matrices(frame_count, filter_indices, filter_matrices):
    filter_matrix_size = len(filter_matrices[0])
    frame_numbers = np.arange(frame_count)
    interpolated_matrices = np.zeros((frame_count, filter_matrix_size))
    for x in range(filter_matrix_size):
        interpolated_matrices[:, x] = np.interp(frame_numbers, filter_indices, filter_matrices[:, x])
    return interpolated_matrices

_FFMPEG_EXECUTABLES = None

def get_ffmpeg_executables():
    """Return the (ffmpeg, ffprobe) executables bundled via static-ffmpeg.

    The binaries are downloaded and cached locally on first use, so no
    system-wide ffmpeg installation is required. Returns (None, None) if the
    executables cannot be obtained.
    """
    global _FFMPEG_EXECUTABLES
    if _FFMPEG_EXECUTABLES is None:
        try:
            from static_ffmpeg import run as static_ffmpeg_run
            _FFMPEG_EXECUTABLES = static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
        except Exception as error:
            print("Could not obtain bundled ffmpeg ({}); output video will not contain audio.".format(error))
            _FFMPEG_EXECUTABLES = (None, None)
    return _FFMPEG_EXECUTABLES

def ensure_ffmpeg_available(progress_callback=None):
    """Eagerly download/locate the bundled ffmpeg binaries.

    static-ffmpeg fetches the platform binaries on first use. Triggering that
    download up front (e.g. at application startup) means the user is told
    immediately if it fails, instead of silently losing audio later on.

    When ``progress_callback`` is provided it is forwarded to
    :class:`ffmpeg_downloader.FfmpegDownloader` and called as
    ``progress_callback(downloaded_bytes, total_bytes)`` while the binaries are
    downloading (not when they are already cached), so a GUI can show progress.

    Returns None on success, or a human-readable error message string describing
    why the binaries could not be obtained.
    """
    global _FFMPEG_EXECUTABLES
    try:
        from ffmpeg_downloader import FfmpegDownloader
        _FFMPEG_EXECUTABLES = FfmpegDownloader(progress_callback).ensure_available()
    except Exception as error:
        return str(error)
    return None

def has_audio_stream(video_path):
    """Detect whether the given video file contains an audio stream.

    Returns True if an audio stream is found, False if none is found, and None
    when it cannot be determined (e.g. ffprobe is not available).
    """
    _, ffprobe = get_ffmpeg_executables()
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() != b""

def mux_audio(corrected_video_path, original_video_path, output_path):
    """Mux the audio track of the original video into the corrected (video-only) file.

    Returns True if the audio was successfully muxed into output_path, False
    otherwise. When False is returned the caller should fall back to using the
    corrected (video-only) file as the output.
    """
    ffmpeg, _ = get_ffmpeg_executables()
    if not ffmpeg:
        print("ffmpeg not available; output video will not contain audio.")
        return False

    # Skip muxing only when we can positively confirm the source has no audio.
    # When detection is unavailable, ffmpeg's optional "1:a?" mapping handles a
    # missing audio track gracefully.
    if has_audio_stream(original_video_path) is False:
        return False

    try:
        result = subprocess.run(
            [
                ffmpeg, "-y",
                "-i", corrected_video_path,
                "-i", original_video_path,
                "-map", "0:v",
                "-map", "1:a?",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print("Failed to run ffmpeg ({}); output video will not contain audio.".format(error))
        return False

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        print(f"ffmpeg failed to mux audio (exit {result.returncode}): {err}")
        return False

    return True

def mux_audio_async(corrected_video_path, original_video_path, output_path):
    """Run mux_audio on a background thread.

    ffmpeg can take a while, and the GUI runs a single-threaded event loop, so
    calling mux_audio() directly would freeze the window. Returning a Future
    lets the caller keep yielding control back to the event loop (so the window
    stays responsive) and apply the result once the work is finished.

    The returned ThreadPoolExecutor is shut down (without waiting) once the
    future completes, so the caller only needs to track the future.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        mux_audio, corrected_video_path, original_video_path, output_path
    )
    # Release the worker thread as soon as the job is done; cancel_futures is
    # unnecessary because the single submitted task is the one we are awaiting.
    future.add_done_callback(lambda _f: executor.shutdown(wait=False))
    return future

def process_video(video_data, yield_preview=False):
    cap = cv2.VideoCapture(video_data["input_video_path"])
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_data["fps"]
    frame_count = video_data["frame_count"]

    output_video_path = video_data["output_video_path"]

    # Write corrected frames to a temporary file first. OpenCV's VideoWriter
    # cannot carry audio, so the original audio is muxed back in afterwards.
    output_dir = os.path.dirname(os.path.abspath(output_video_path))
    temp_fd, temp_video_path = tempfile.mkstemp(suffix=".mp4", dir=output_dir)
    os.close(temp_fd)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_video = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
    if not new_video.isOpened():
        cap.release()
        new_video.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise RuntimeError("Failed to open VideoWriter — check codec and output path")

    try:
        # Precompute interpolated filter matrices
        print("Precomputing filter matrices...")
        interpolated_matrices = precompute_filter_matrices(
            frame_count, video_data["filter_indices"], np.array(video_data["filters"])
        )

        print("Processing...")
        count = 0
        while cap.isOpened():
            count += 1
            percent = 100 * count / frame_count
            print("{:.2f}%".format(percent), end="\r")
            ret, frame = cap.read()

            if not ret:
                # End video read if we have gone beyond reported frame count
                if count >= frame_count:
                    break

                # Failsafe to prevent an infinite loop
                if count >= 1e6:
                    break

                # Otherwise this is just a faulty frame read, try reading next
                continue

            # Apply the filter using precomputed matrix
            rgb_mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            corrected_mat = apply_filter(rgb_mat, interpolated_matrices[count - 1])
            corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)
            new_video.write(corrected_mat)

            if yield_preview:
                preview = frame.copy()
                width = preview.shape[1] // 2
                height = preview.shape[0] // 2
                preview[:, width:] = corrected_mat[:, width:]

                preview = cv2.resize(preview, (width, height))

                yield percent, cv2.imencode('.png', preview)[1].tobytes()
            else:
                yield None

        cap.release()
        new_video.release()

        # Mux the original audio back into the corrected video on a background
        # thread. ffmpeg is blocking, so running it inline would freeze the
        # single-threaded GUI event loop. Instead we kick it off asynchronously
        # and keep yielding control back to the caller (the GUI advances this
        # generator one step per loop iteration), so the window stays responsive
        # while ffmpeg runs. If muxing is not possible (no ffmpeg, no audio
        # track, or an ffmpeg error) we fall back to the video-only file.
        ffmpeg, _ = get_ffmpeg_executables()
        if not ffmpeg:
            # ffmpeg could not be obtained, so the corrected (video-only) file
            # is the best we can produce. Surface this to the GUI so the user
            # knows why their output has no audio.
            print("ffmpeg not found; output video will not contain audio.")
            os.replace(temp_video_path, output_video_path)
            if yield_preview:
                yield "ffmpeg not found - saved video without audio", None
            else:
                yield None
            return

        mux_future = mux_audio_async(
            temp_video_path, video_data["input_video_path"], output_video_path
        )

        mux_started_at = time.monotonic()
        while not mux_future.done():
            if yield_preview:
                elapsed = int(time.monotonic() - mux_started_at)
                # Animate the status so it is obvious the app is still alive.
                dots = "." * (1 + elapsed % 3)
                yield "Muxing audio{} ({}s)".format(dots, elapsed), None
            else:
                yield None
            # Yield the GIL briefly so the worker thread makes progress and we
            # do not spin too tightly between event-loop iterations.
            time.sleep(0.05)

        if mux_future.result():
            os.remove(temp_video_path)
        else:
            os.replace(temp_video_path, output_video_path)
    finally:
        cap.release()
        new_video.release()
        # Remove the temporary file if it is still around (e.g. after an error).
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage")
        print("-"*20)
        print("For image:")
        print("$python correct.py image <source_image_path> <output_image_path>\n")
        print("-"*20)
        print("For video:")
        print("$python correct.py video <source_video_path> <output_video_path>\n")
        exit(0)

    if (sys.argv[1]) == "image":
        mat = cv2.imread(sys.argv[2])
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        
        corrected_mat = correct(mat)

        cv2.imwrite(sys.argv[3], corrected_mat)
    
    else:

        for item in analyze_video(sys.argv[2], sys.argv[3]):

            if type(item) == dict:
                video_data = item
            
        [x for x in process_video(video_data, yield_preview=False)]
        
