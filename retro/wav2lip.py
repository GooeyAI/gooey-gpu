import mimetypes
import os
import subprocess
import sys
import typing
from tempfile import TemporaryDirectory

import requests
from celery.signals import worker_init
from kombu import Queue
from pydantic import Field, BaseModel

import gooey_gpu
from api import PipelineInfo
from celeryconfig import app

sys.path.append(os.path.join(os.path.dirname(__file__), "wav2lip-src"))
print(os.path.join(os.path.dirname(__file__), "wav2lip-src"))

import inference

QUEUE_PREFIX = os.environ.get("QUEUE_PREFIX", "gooey-gpu")
MODEL_IDS = os.environ["WAV2LIP_MODEL_IDS"].split()

app.conf.task_queues = app.conf.task_queues or []
for model_id in MODEL_IDS:
    queue = os.path.join(QUEUE_PREFIX, model_id).strip("/")
    app.conf.task_queues.append(Queue(queue))


@worker_init.connect()
def init(**kwargs):
    # app.conf.task_queues = []
    for model_id in MODEL_IDS:
        inference.do_load(os.path.join(gooey_gpu.CHECKPOINTS_DIR, model_id))


class Wav2LipInputs(BaseModel):
    face: str = Field(
        description="video/image that contains faces to use",
    )
    audio: str = Field(
        description="video/audio file to use as raw audio source",
    )
    pads: typing.List[int] = Field(
        description="Padding for the detected face bounding box.\n"
        "Please adjust to include chin at least\n"
        'Format: "top bottom left right"',
        default=[0, 10, 0, 0],
    )
    smooth: bool = Field(
        description="Smooth face detections over a short temporal window",
        default=True,
    )
    fps: float = Field(
        description="Can be specified only if input is a static image",
        default=25.0,
    )
    out_height: int = Field(
        description="Output video height. Best results are obtained at 480 or 720",
        default=480,
    )


@app.task(name="wav2lip")
@gooey_gpu.endpoint
def wav2lip(pipeline: PipelineInfo, inputs: Wav2LipInputs):
    face_mime_type = mimetypes.guess_type(inputs.face)[0] or ""
    if not ("video/" in face_mime_type or "image/" in face_mime_type):
        raise ValueError(f"Unsupported face format {face_mime_type!r}")

    audio_mime_type = mimetypes.guess_type(inputs.audio)[0] or ""
    if not ("audio/" in audio_mime_type or "video/" in audio_mime_type):
        raise ValueError(f"Unsupported audio format {audio_mime_type!r}")

    with TemporaryDirectory() as tmpdir:
        face_path = os.path.join(tmpdir, "face" + os.path.splitext(inputs.face)[1])
        audio_path = os.path.join(tmpdir, "audio" + os.path.splitext(inputs.audio)[1])
        result_path = os.path.join(tmpdir, "result_voice.mp4")

        r = requests.get(inputs.face, allow_redirects=True)
        r.raise_for_status()
        with open(face_path, "wb") as f:
            f.write(r.content)

        r = requests.get(inputs.audio, allow_redirects=True)
        r.raise_for_status()
        with open(audio_path, "wb") as f:
            f.write(r.content)

        if audio_mime_type != "audio/wav":
            wav_audio_path = audio_path + ".wav"
            args = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                wav_audio_path,
            ]  # fmt:skip
            print("\t$ " + " ".join(args))
            print(subprocess.check_output(args, encoding="utf-8"))
            audio_path = wav_audio_path

        args = [
            "--checkpoint_path", os.path.join(gooey_gpu.CHECKPOINTS_DIR, pipeline.model_id),
            "--face", face_path,
            "--audio", audio_path,
            "--pads", *inputs.pads,
            "--fps", inputs.fps,
            "--out_height", inputs.out_height,
            "--outfile", result_path,
        ]  # fmt:skip
        if not inputs.smooth:
            args += ["--nosmooth"]
        args = list(map(str, args))
        print("\t$ inference.py " + " ".join(args))
        inference.args = inference.parser.parse_args(args)

        try:
            inference.main()
        except ValueError as e:
            print(f"-> Encountered error, skipping lipsync: {e}")
            args = [
                "ffmpeg",
                "-y",
                # "-vsync", "0", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                "-stream_loop", "-1",
                "-i", face_path,
                "-i", audio_path,
                "-shortest",
                "-fflags", "+shortest",
                "-max_interleave_delta", "100M",
                "-map", "0:v:0",
                "-map", "1:a:0",
                # "-c", "copy",
                # "-c:v", "h264_nvenc",
                result_path,
            ]  # fmt:skip
            args = list(map(str, args))
            print("\t$ " + " ".join(args))
            print(subprocess.check_output(args, encoding="utf-8"))

        with open(result_path, "rb") as f:
            for url in pipeline.upload_urls:
                # upload to given url
                r = requests.put(url, headers={"Content-Type": "video/mp4"}, data=f)
                r.raise_for_status()
