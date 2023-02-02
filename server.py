import traceback
import typing
from collections import defaultdict

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DiffusionPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline,
)
from fastapi import FastAPI
from pydantic import BaseModel

import gooey_gpu

app = FastAPI()

MAX_IMAGE_SIZE = (768, 768)


class BaseInputs(BaseModel):
    prompt: typing.List[str]
    negative_prompt: typing.List[str] = None
    num_images_per_prompt: int
    num_inference_steps: int
    guidance_scale: float


class PipelineInfo(BaseModel):
    upload_urls: typing.List[str]
    model_id: str
    scheduler: str = None
    seed: int


class Text2ImgInputs(BaseInputs):
    width: int
    height: int


@app.post("/text2img/")
@gooey_gpu.endpoint
def text2img(pipeline: PipelineInfo, inputs: Text2ImgInputs):
    return predict(
        pipeline,
        inputs,
        StableDiffusionPipeline,
    )


class Img2ImgInputs(BaseInputs):
    image: typing.List[str]
    strength: float


@app.post("/img2img/")
@gooey_gpu.endpoint
def img2img(pipeline: PipelineInfo, inputs: Img2ImgInputs):
    return predict(
        pipeline,
        inputs,
        StableDiffusionImg2ImgPipeline,
        image=gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE),
    )


class InpaintInputs(BaseInputs):
    image: typing.List[str]
    mask_image: typing.List[str]


@app.post("/inpaint/")
@gooey_gpu.endpoint
def inpaint(pipeline: PipelineInfo, inputs: InpaintInputs):
    return predict(
        pipeline,
        inputs,
        StableDiffusionInpaintPipeline,
        image=gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE),
        mask_image=gooey_gpu.download_images(inputs.mask_image, MAX_IMAGE_SIZE),
    )


class UpscaleInputs(BaseInputs):
    image: typing.List[str]


@app.post("/upscale/")
@gooey_gpu.endpoint
def upscale(pipeline: PipelineInfo, inputs: UpscaleInputs):
    return predict(
        pipeline,
        inputs,
        StableDiffusionUpscalePipeline,
        image=gooey_gpu.download_images(inputs.image, (512, 512)),
    )


class InstructPix2PixInputs(BaseInputs):
    image: typing.List[str]
    image_guidance_scale: float


@app.post("/instruct_pix2pix/")
@gooey_gpu.endpoint
def instruct_pix2pix(pipeline: PipelineInfo, inputs: InstructPix2PixInputs):
    return predict(
        pipeline,
        inputs,
        StableDiffusionInstructPix2PixPipeline,
        image=gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE),
    )


def predict(
    pipeline: PipelineInfo,
    inputs: BaseInputs,
    pipe_cls,
    **kwargs,
):
    inputs_dict = inputs.dict()
    inputs_dict.update(kwargs)

    with gooey_gpu.gpu_lock():
        pipe = load_pipeline(pipeline, pipe_cls)

        with gooey_gpu.use_gpu(pipe):
            _remove_safety_checker(pipe)
            pipe.enable_xformers_memory_efficient_attention()

            generator = torch.Generator("cuda").manual_seed(pipeline.seed)
            output = pipe(**inputs_dict, generator=generator)

            output_images = output.images

    gooey_gpu.upload_images(output_images, pipeline.upload_urls)


pipes_cache = defaultdict(dict)
schedulers_cache = defaultdict(dict)


def load_pipeline(
    pipeline: PipelineInfo,
    pipe_cls,
):
    pipes = pipes_cache[pipe_cls.__name__]

    try:
        pipe = pipes[pipeline.model_id]
    except KeyError:
        pipe = pipe_cls.from_pretrained(pipeline.model_id, torch_dtype=torch.float16)
        pipes[pipeline.model_id] = pipe
        update_schedulers(pipeline.model_id, pipe)

    try:
        pipe.schduler = schedulers_cache[pipeline.model_id][pipeline.scheduler]
    except KeyError:
        raise ValueError(
            f"Incompatible scheduler `{pipeline.scheduler}` for `{pipeline.model_id}`"
        )

    return pipe


def update_schedulers(model_id: str, pipe: DiffusionPipeline):
    schedulers = {None: pipe.scheduler}
    for cls in pipe.scheduler.compatibles:
        try:
            schedulers[cls.__name__] = cls.from_config(pipe.scheduler.config)
        except ImportError:
            traceback.print_exc()
            continue
    schedulers_cache[model_id] = schedulers


def _remove_safety_checker(pipe):
    """If there's an nsfw filter, replace it with a dummy"""
    try:
        if pipe.safety_checker:
            pipe.safety_checker = _dummy
    except AttributeError:
        pass


def _dummy(images, **kwargs):
    return images, False
