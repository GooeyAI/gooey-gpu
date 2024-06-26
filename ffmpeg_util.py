import json
import subprocess
import typing
from fractions import Fraction

import numpy as np
from pydantic import BaseModel


class VideoMetadata(BaseModel):
    width: int = 0
    height: int = 0
    num_frames: int = 0
    duration_sec: float = 0
    fps: typing.Optional[float] = None
    codec_name: typing.Optional[str] = None


class InputOutputVideoMetadata(BaseModel):
    input: VideoMetadata
    output: VideoMetadata


def ffprobe_video(input_path: str) -> VideoMetadata:
    cmd_args = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams", input_path,
        "-select_streams", "v:0",
    ]  # fmt:skip
    print("\t$ " + " ".join(cmd_args))
    data = json.loads(subprocess.check_output(cmd_args, text=True))

    try:
        stream = data["streams"][0]
    except IndexError:
        raise ValueError("input has no video streams")

    try:
        fps = float(Fraction(stream["avg_frame_rate"]))
    except ZeroDivisionError:
        fps = None

    metadata = VideoMetadata(
        width=int(stream["width"]) // 2 * 2,
        height=int(stream["height"]) // 2 * 2,
        num_frames=int(stream.get("nb_frames") or 0),
        duration_sec=float(stream.get("duration") or 0),
        fps=fps,
        codec_name=stream.get("codec_name"),
    )
    print(repr(metadata))
    return metadata


def ffmpeg_read_input_frames(
    *, width: float, height: float, input_path: str, fps: float
) -> typing.Iterator[np.ndarray]:
    cmd_args = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", input_path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "pipe:1",
    ]  # fmt:skip
    print("\t$ " + " ".join(cmd_args))
    ffproc = subprocess.Popen(cmd_args, stdout=subprocess.PIPE)

    while True:
        im_bytes = ffproc.stdout.read(height * width * 3)
        if not im_bytes:
            break
        im_cv2 = np.frombuffer(im_bytes, dtype=np.uint8).reshape((height, width, 3))
        yield im_cv2


def ffmpeg_get_writer_proc(
    *, width: int, height: int, output_path: str, fps: float, audio_path: str
) -> subprocess.Popen:
    cmd_args = [
        "ffmpeg", "-hide_banner", "-nostats",
        # "-thread_queue_size", "128",
        "-pixel_format", "rgb24",
        "-f", "rawvideo",
        # "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",  # stdin
        "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        # "-c:a", "copy",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p", # because iphone, see https://trac.ffmpeg.org/wiki/Encode/H.264#Encodingfordumbplayers
        # "-preset", "ultrafast",
        output_path,
    ]  # fmt:skip
    print("\t$ " + " ".join(cmd_args))
    return subprocess.Popen(cmd_args, stdin=subprocess.PIPE)
