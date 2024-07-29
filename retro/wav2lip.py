import mimetypes
import os
import subprocess
import sys
import typing
from functools import lru_cache
from tempfile import TemporaryDirectory

import PIL.Image
import cv2
import numpy as np
import requests
import torch
from batch_face import RetinaFace
from pydantic import Field, BaseModel
from tqdm import tqdm

import gooey_gpu
from api import PipelineInfo
from celeryconfig import app, setup_queues
from retro.Wav2Lip.models import Wav2Lip

sys.path.append(os.path.join(os.path.dirname(__file__), "Wav2Lip"))

from retro.Wav2Lip import audio

MAX_RES = 1920 * 1080
mel_step_size = 16


@lru_cache
def load_models(model_id):
    checkpoint_path = os.path.join(gooey_gpu.CHECKPOINTS_DIR, model_id)
    model = load_model(checkpoint_path)

    detector = RetinaFace(
        gpu_id=0,
        model_path=os.path.join(gooey_gpu.CHECKPOINTS_DIR, "mobilenet.pth"),
        network="mobilenet",
    )

    return model, detector


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)

    model = model.to(gooey_gpu.DEVICE_ID)
    return model.eval()


model_ids = os.environ["WAV2LIP_MODEL_IDS"].split()
setup_queues(model_ids=model_ids, load_fn=load_models)


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
    fps: float = Field(
        description="Can be specified only if input is a static image",
        default=25.0,
    )
    out_height: int = Field(
        description="Output video height. Best results are obtained at 480 or 720",
        default=480,
    )
    batch_size: int = 256


@app.task(name="wav2lip")
@gooey_gpu.endpoint
def wav2lip(pipeline: PipelineInfo, inputs: Wav2LipInputs):
    model, detector = load_models(pipeline.model_id)

    with TemporaryDirectory() as tmpdir:
        audio_ext = os.path.splitext(inputs.audio)[1]
        audio_path = os.path.join(tmpdir, "audio" + audio_ext)
        gooey_gpu.download_file_to_path(url=inputs.audio, path=audio_path)
        input_audio_metadata = gooey_gpu.ffprobe_audio(audio_path)
        # make sure audio is not 0 seconds
        if input_audio_metadata.duration_sec <= 0.1:
            raise gooey_gpu.UserError("Audio is too short")
        # convert audio to wav
        if input_audio_metadata.codec_name != "pcm_s16le":
            wav_audio_path = audio_path + ".wav"
            gooey_gpu.ffmpeg(
                "-i", audio_path,
                "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                wav_audio_path,
            )  # fmt:skip
            audio_path = wav_audio_path

        video_ext = os.path.splitext(inputs.face)[1]
        video_path = os.path.join(tmpdir, "video" + video_ext)
        gooey_gpu.download_file_to_path(url=inputs.face, path=video_path)
        input_video_metadata = gooey_gpu.ffprobe_video(video_path)

        inputs.audio = audio_path
        inputs.face = video_path
        result_path = os.path.join(tmpdir, "result_voice.mp4")

        try:
            main(model=model, detector=detector, outfile=result_path, inputs=inputs)
        except FaceNotFoundException as e:
            print(f"-> Encountered error, skipping lipsync: {e}")
            gooey_gpu.ffmpeg(
                # "-vsync", "0", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                "-stream_loop", "-1",
                "-i", video_path,
                "-i", audio_path,
                "-shortest",
                "-fflags", "+shortest",
                "-max_interleave_delta", "100M",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-pix_fmt", "yuv420p",
                # "-c", "copy",
                # "-c:v", "h264_nvenc",
                result_path,
            )  # fmt:skip

        with open(result_path, "rb") as f:
            for url in pipeline.upload_urls:
                # upload to given url
                r = requests.put(url, headers={"Content-Type": "video/mp4"}, data=f)
                gooey_gpu.raise_for_status(r)

        return gooey_gpu.InputOutputVideoMetadata(
            input=input_video_metadata, output=gooey_gpu.ffprobe_video(result_path)
        )


def main(model, detector, outfile: str, inputs: Wav2LipInputs):
    lip_size = 96

    face_mime_type = mimetypes.guess_type(inputs.face)[0] or ""
    is_static = "image/" in face_mime_type and "image/gif" not in face_mime_type

    if not os.path.isfile(inputs.face):
        raise ValueError("--face argument must be a valid path to video/image file")

    if is_static:
        input_stream = None
        fps = inputs.fps
        frame = cv2.cvtColor(
            np.array(PIL.Image.open(inputs.face).convert("RGB")), cv2.COLOR_RGB2BGR
        )
        frame = resize_frame(frame, inputs.out_height)
    else:
        input_stream = cv2.VideoCapture(inputs.face)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame = None

    ffproc = None
    prev_faces = None

    mel_chunks = get_mel_chunks(inputs.audio, fps)
    for idx in tqdm(range(0, len(mel_chunks), inputs.batch_size)):
        if is_static:
            frame_batch = [frame.copy()] * inputs.batch_size
        else:
            frame_batch = list(
                read_n_frames(
                    input_stream, inputs.face, inputs.batch_size, inputs.out_height
                )
            )

        if idx == 0:
            frame_h, frame_w = frame_batch[0].shape[:-1]
            gooey_gpu.ffmpeg(
                # "-thread_queue_size", "128",
                "-pixel_format", "bgr24", # to match opencv
                "-f", "rawvideo",
                # "-vcodec", "rawvideo",
                "-s", f"{frame_w}x{frame_h}",
                "-r", str(fps),
                "-i", "pipe:0", # stdin
                "-i", inputs.audio,
                # "-vcodec", "libx264",
                "-pix_fmt", "yuv420p", # because iphone, see https://trac.ffmpeg.org/wiki/Encode/H.264#Encodingfordumbplayers
                # "-preset", "ultrafast",
                outfile,
            )  # fmt:skip

        mel_batch = mel_chunks[idx : idx + inputs.batch_size]
        frame_batch = frame_batch[: len(mel_batch)]

        coords_batch, prev_faces = face_detect(
            detector, frame_batch, inputs.pads, prev_faces
        )
        img_batch = [
            cv2.resize(image[y1:y2, x1:x2], (lip_size, lip_size))
            for image, (x1, y1, x2, y2) in zip(frame_batch, coords_batch)
        ]

        img_batch = np.asarray(img_batch)
        mel_batch = np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, lip_size // 2 :] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(
            gooey_gpu.DEVICE_ID
        )
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(
            gooey_gpu.DEVICE_ID
        )
        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c in zip(pred, frame_batch, coords_batch):
            x1, y1, x2, y2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            cv2.imwrite(f"{outfile}_{idx}.png", f)
            ffproc.stdin.write(f.tostring())

    if input_stream:
        input_stream.release()
    if ffproc:
        ffproc.stdin.close()
        ffproc.wait()


def read_n_frames(
    video_stream, face: str, num_frames: int, out_height: int
) -> typing.Generator[np.ndarray, None, None]:
    for _ in range(num_frames):
        ret, frame = video_stream.read()

        if not ret:
            video_stream.release()
            video_stream = cv2.VideoCapture(face)
            ret, frame = video_stream.read()
            if not ret:
                raise gooey_gpu.UserError("Video file contains no frames")

        frame = resize_frame(frame, out_height)
        yield frame


def resize_frame(frame, out_height: int) -> np.ndarray:
    if frame.shape[0] * frame.shape[1] > MAX_RES:
        raise gooey_gpu.UserError(
            "Input video resolution exceeds 1920x1080. Please downscale to 1080p."
        )
    aspect_ratio = frame.shape[1] / frame.shape[0]
    out_width = int(out_height * aspect_ratio)
    if out_width % 2 != 0:
        out_width -= 1
    frame = cv2.resize(frame, (out_width, out_height))
    return frame


def get_mel_chunks(audio_path: str, fps: float) -> typing.List[np.ndarray]:
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    print(f"{mel.shape=}")

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
        )

    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print(f"{len(mel_chunks)=}")

    return mel_chunks


def face_detect(
    detector, images, pads: typing.List[int], prev_faces: list
) -> typing.Tuple[np.ndarray, list]:
    results = []
    pady1, pady2, padx1, padx2 = pads

    for image, faces in zip(images, detector(images)):
        if not (faces or prev_faces):
            raise FaceNotFoundException(
                "Face not detected! Ensure the video contains a face in all the frames."
            )
        faces = faces or prev_faces
        prev_faces = faces

        box, landmarks, score = faces[0]
        rect = tuple(map(int, box))

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)

    return boxes, prev_faces


class FaceNotFoundException(ValueError):
    pass
