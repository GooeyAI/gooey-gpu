import json
import subprocess
import typing
from fractions import Fraction

import numpy as np
from pydantic import BaseModel

from exceptions import UserError


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


class AudioMetadata(BaseModel):
    duration_sec: float = 0
    codec_name: typing.Optional[str] = None


FFMPEG_ERR_MSG = (
    "Unsupported File Format\n\n"
    "We encountered an issue processing your file as it appears to be in a format not supported by our system or may be corrupted. "
    "You can find a list of supported formats at [FFmpeg Formats](https://ffmpeg.org/general.html#File-Formats)."
)


def ffprobe_audio(input_path: str) -> AudioMetadata:
    text = call_cmd(
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams", input_path,
        "-select_streams", "a:0",
        err_msg=FFMPEG_ERR_MSG,
    )  # fmt:skip
    data = json.loads(text)

    try:
        stream = data["streams"][0]
    except IndexError:
        raise UserError(
            "Input has no audio streams. Make sure the you have uploaded an appropriate audio/video file."
        )

    return AudioMetadata(
        duration_sec=float(stream.get("duration") or 0),
        codec_name=stream.get("codec_name"),
    )


def ffprobe_video(input_path: str) -> VideoMetadata:
    text = call_cmd(
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams", input_path,
        "-select_streams", "v:0",
        err_msg=FFMPEG_ERR_MSG,
    )  # fmt:skip
    data = json.loads(text)

    try:
        stream = data["streams"][0]
    except IndexError:
        raise UserError(
            "Input has no video streams. Make sure the video you have uploaded is not corrupted."
        )

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
    *,
    width: float,
    height: float,
    input_path: str,
    fps: float,
    pixel_format: str = "rgb24",
) -> typing.Iterator[np.ndarray]:
    cmd_args = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", input_path,
        "-f", "rawvideo",
        "-pix_fmt", pixel_format,
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "pipe:1",
    ]  # fmt:skip
    print("\t$ " + " ".join(cmd_args))
    ffproc = subprocess.Popen(cmd_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    while True:
        retcode = ffproc.poll()
        if retcode is not None:
            output = ffproc.stderr.read().decode()
            err = subprocess.SubprocessError(output)
            err.__cause__ = subprocess.CalledProcessError(retcode, ffproc.args, output)
            raise UserError(FFMPEG_ERR_MSG) from err
        im_bytes = ffproc.stdout.read(height * width * 3)
        if not im_bytes:
            break
        im_cv2 = np.frombuffer(im_bytes, dtype=np.uint8).reshape((height, width, 3))
        yield im_cv2


def ffmpeg_get_writer_proc(
    *,
    width: int,
    height: int,
    output_path: str,
    fps: float,
    audio_path: str,
    pixel_format: str = "rgb24",
) -> subprocess.Popen:
    cmd_args = [
        "ffmpeg", "-hide_banner", "-nostats",
        # "-thread_queue_size", "128",
        "-pixel_format", pixel_format,
        "-f", "rawvideo",
        # "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",  # stdin
        "-i", audio_path,
        "-map", "0:v", "-map", "1:a", "-shortest",
        # "-c:a", "copy",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p", # because iphone, see https://trac.ffmpeg.org/wiki/Encode/H.264#Encodingfordumbplayers
        # "-preset", "ultrafast",
        output_path,
    ]  # fmt:skip
    print("\t$ " + " ".join(cmd_args))
    return subprocess.Popen(
        cmd_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def ensure_img_even_dimensions(img: np.ndarray) -> np.ndarray:
    """make sure the image dimensions are divisble by 2 for ffmpeg libx264"""
    return img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]


def ffmpeg_write_output_frame(ffproc: subprocess.Popen, img: np.ndarray):
    retcode = ffproc.poll()
    if retcode is not None:
        output = ffproc.stdout.read().decode()
        err = subprocess.SubprocessError(output)
        err.__cause__ = subprocess.CalledProcessError(retcode, ffproc.args, output)
        raise UserError(FFMPEG_ERR_MSG) from err
    ffproc.stdin.write(img.tostring())


def ffmpeg(*args) -> str:
    return call_cmd("ffmpeg", "-hide_banner", "-y", *args, err_msg=FFMPEG_ERR_MSG)


def call_cmd(
    *args, err_msg: str = "", ok_returncodes: typing.Iterable[int] = ()
) -> str:
    print("\t$ " + " ".join(map(str, args)))
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        if e.returncode in ok_returncodes:
            return e.output
        err_msg = err_msg or f"{str(args[0]).capitalize()} Error"
        try:
            raise subprocess.SubprocessError(e.output) from e
        except subprocess.SubprocessError as e:
            raise UserError(err_msg) from e
