import math
import os
import typing
from functools import lru_cache
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import PIL.Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from pydantic import BaseModel, HttpUrl
from realesrgan import RealESRGANer

import gooey_gpu
from celeryconfig import app, setup_queues
from ffmpeg_util import (
    ffmpeg_get_writer_proc,
    ffmpeg_read_input_frames,
    ffprobe_video,
    VideoMetadata,
    InputOutputVideoMetadata,
)

MAX_RES = 1920 * 1080


class EsrganPipeline(BaseModel):
    upload_urls: typing.List[HttpUrl]  # upload url for the output video
    model_id: str


class EsrganInputs(BaseModel):
    image: typing.Optional[str]
    video: typing.Optional[str]
    scale: float = 2


@app.task(name="realesrgan")
@gooey_gpu.endpoint
def realesrgan(
    pipeline: EsrganPipeline, inputs: EsrganInputs
) -> InputOutputVideoMetadata:
    esrganer = load_esrgan_model(pipeline.model_id)

    def enhance(frame, outscale_factor):
        restored_img, _ = esrganer.enhance(frame, outscale=outscale_factor)
        return restored_img

    return run_enhancer(
        image=inputs.image,
        video=inputs.video,
        scale=inputs.scale,
        upload_url=pipeline.upload_urls[0],
        enhance=enhance,
    )


class GfpganPipeline(BaseModel):
    upload_urls: typing.List[HttpUrl]  # upload url for the output video
    model_id: str
    bg_model_id: typing.Optional[str] = None


class GfpganInputs(BaseModel):
    image: typing.Optional[str]
    video: typing.Optional[str]
    scale: float = 2
    weight: float = 0.5


@app.task(name="gfpgan")
@gooey_gpu.endpoint
def gfpgan(pipeline: GfpganPipeline, inputs: GfpganInputs) -> InputOutputVideoMetadata:
    gfpganer = load_gfpgan_model(pipeline.model_id)
    if pipeline.bg_model_id:
        gfpganer.bg_upsampler = load_esrgan_model(pipeline.bg_model_id)

    def enhance(frame, upscale_factor):
        gfpganer.upscale = gfpganer.face_helper.upscale_factor = upscale_factor
        cropped_faces, restored_faces, restored_img = gfpganer.enhance(
            frame,
            # has_aligned=args.aligned,
            # only_center_face=args.only_center_face,
            # paste_back=True,
            weight=inputs.weight,
        )
        return restored_img

    return run_enhancer(
        image=inputs.image,
        video=inputs.video,
        scale=inputs.scale,
        upload_url=pipeline.upload_urls[0],
        enhance=enhance,
    )


def run_enhancer(
    image: str | None,
    video: str | None,
    scale: float,
    upload_url: str,
    enhance: typing.Callable,
) -> InputOutputVideoMetadata:
    input_file = image or video
    assert input_file, "Please provide an image or video input"

    with TemporaryDirectory() as save_dir:
        input_path, _ = urlretrieve(
            input_file,
            os.path.join(save_dir, "input" + os.path.splitext(input_file)[1]),
        )
        output_path = os.path.join(save_dir, "out.mp4")

        response = InputOutputVideoMetadata(
            input=ffprobe_video(input_path), output=VideoMetadata()
        )
        # ensure max input/output is 1080p
        input_pixels = response.input.width * response.input.height
        if input_pixels > MAX_RES:
            raise ValueError(
                "Input video resolution exceeds 1920x1080. Please downscale to 1080p."
            )
        max_scale = math.sqrt(MAX_RES / input_pixels)
        upscale_factor = max(min(scale, max_scale), 1)
        print(f"Using upscale factor: {upscale_factor}")

        ffproc = None
        for frame in ffmpeg_read_input_frames(
            width=response.input.width,
            height=response.input.height,
            input_path=input_path,
            fps=response.input.fps or 24,
        ):
            restored_img = enhance(frame, upscale_factor)
            if restored_img is None:
                continue

            if image:
                gooey_gpu.upload_image(
                    PIL.Image.fromarray(restored_img, mode="RGB"),
                    upload_url,
                )
                response.output.codec_name = "png"
                break

            if ffproc is None:
                response.output.width = restored_img.shape[1]
                response.output.height = restored_img.shape[0]
                response.output.fps = response.input.fps or 24
                ffproc = ffmpeg_get_writer_proc(
                    width=response.output.width,
                    height=response.output.height,
                    fps=response.output.fps,
                    output_path=output_path,
                    audio_path=input_path,
                )
            ffproc.stdin.write(restored_img.tostring())
            response.output.num_frames += 1

        if ffproc is not None:
            ffproc.stdin.close()
            ffproc.wait()
            with open(output_path, "rb") as f:
                gooey_gpu.upload_video_from_bytes(f.read(), upload_url)

        if response.output.num_frames:
            response.output.duration_sec = (
                response.output.num_frames / response.output.fps
            )
            response.output.codec_name = "h264"

    return response


@lru_cache
def load_gfpgan_model(model_id: str) -> "GFPGANer":
    # from https://github.com/TencentARC/GFPGAN/blob/7552a7791caad982045a7bbe5634bbf1cd5c8679/inference_gfpgan.py#L82
    if model_id == "GFPGANv1":
        arch = "original"
        channel_multiplier = 1
        url = (
            "https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth"
        )
    elif model_id == "GFPGANCleanv1-NoCE-C2":
        arch = "clean"
        channel_multiplier = 2
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth"
    elif model_id == "GFPGANv1.3":
        arch = "clean"
        channel_multiplier = 2
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    elif model_id == "GFPGANv1.4":
        arch = "clean"
        channel_multiplier = 2
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    elif model_id == "RestoreFormer":
        arch = "RestoreFormer"
        channel_multiplier = 2
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    else:
        raise ValueError(f"Model {model_id} not found")

    gfpgan_checkpoint_dir = os.path.join(gooey_gpu.CHECKPOINTS_DIR, "gfpgan")
    os.makedirs(gfpgan_checkpoint_dir, exist_ok=True)
    try:
        os.symlink(gfpgan_checkpoint_dir, "gfpgan", target_is_directory=True)
    except FileExistsError:
        pass

    print(f"loading {model_id} via {url}...")
    model_path = os.path.join(gfpgan_checkpoint_dir, os.path.basename(url))
    gooey_gpu.download_file_cached(url=url, path=model_path)

    return GFPGANer(
        model_path=model_path,
        # upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        # bg_upsampler=bg_upsampler,
    )


@lru_cache
def load_esrgan_model(model_id: str) -> "RealESRGANer":
    # from https://github.com/xinntao/Real-ESRGAN/blob/a4abfb2979a7bbff3f69f58f58ae324608821e27/inference_realesrgan_video.py#L176
    if model_id == "RealESRGAN_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ]
    elif model_id == "RealESRNet_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        ]
    elif model_id == "RealESRGAN_x4plus_anime_6B":  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]
    elif model_id == "RealESRGAN_x2plus":  # x2 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ]
    else:
        raise ValueError(f"Model {model_id} not found")

    model_path = None
    for url in file_url:
        print(f"loading {model_id} via {url}...")
        model_path = os.path.join(gooey_gpu.CHECKPOINTS_DIR, os.path.basename(url))
        gooey_gpu.download_file_cached(url=url, path=model_path)
    assert model_path, f"Model {model_id} not found"

    return RealESRGANer(
        scale=netscale,
        model_path=model_path,
        # dni_weight=dni_weight,
        model=model,
        # tile=args.tile,
        # tile_pad=args.tile_pad,
        # pre_pad=args.pre_pad,
        half=True,
        device=gooey_gpu.DEVICE_ID,
    )


setup_queues(
    model_ids=os.environ["GFPGAN_MODEL_IDS"].split(),
    load_fn=load_gfpgan_model,
)
setup_queues(
    model_ids=os.environ["ESRGAN_MODEL_IDS"].split(),
    load_fn=load_esrgan_model,
)
