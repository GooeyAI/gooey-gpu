import os
from functools import lru_cache

import torch
from celery.signals import worker_init
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DiffusionPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionControlNetPipeline,
)
from kombu import Queue

import gooey_gpu
from api import (
    PipelineInfo,
    Img2ImgInputs,
    Text2ImgInputs,
    MAX_IMAGE_SIZE,
    InpaintInputs,
    UpscaleInputs,
    InstructPix2PixInputs,
    DiffusersInputs,
)
from celeryconfig import app

QUEUE_PREFIX = os.environ.get("QUEUE_PREFIX", "gooey-gpu")
MODEL_IDS = os.environ["SD_MODEL_IDS"].split()

app.conf.task_queues = app.conf.task_queues or []
for model_id in MODEL_IDS:
    queue = os.path.join(QUEUE_PREFIX, model_id).strip("/")
    app.conf.task_queues.append(Queue(queue))


@worker_init.connect()
def init(**kwargs):
    # app.conf.task_queues = []
    for model_id in MODEL_IDS:
        _load_pipe_cached(StableDiffusionPipeline, model_id)


@app.task(name="diffusion.text2img")
@gooey_gpu.endpoint
def text2img(pipeline: PipelineInfo, inputs: Text2ImgInputs):
    return predict_and_upload(
        pipe_cls=StableDiffusionPipeline,
        pipeline=pipeline,
        inputs=inputs,
    )


@app.task(name="diffusion.img2img")
@gooey_gpu.endpoint
def img2img(pipeline: PipelineInfo, inputs: Img2ImgInputs):
    return predict_and_upload(
        pipe_cls=StableDiffusionImg2ImgPipeline,
        pipeline=pipeline,
        inputs=inputs,
        inputs_extra=dict(
            image=gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE),
        ),
    )


@app.task(name="diffusion.inpaint")
@gooey_gpu.endpoint
def inpaint(pipeline: PipelineInfo, inputs: InpaintInputs):
    image = gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE)
    return predict_and_upload(
        pipe_cls=StableDiffusionInpaintPipeline,
        pipeline=pipeline,
        inputs=inputs,
        inputs_extra=dict(
            image=image,
            mask_image=gooey_gpu.download_images(inputs.mask_image, MAX_IMAGE_SIZE),
            width=image[0].width,
            height=image[0].height,
        ),
    )


@app.task(name="diffusion.upscale")
@gooey_gpu.endpoint
def upscale(pipeline: PipelineInfo, inputs: UpscaleInputs):
    return predict_and_upload(
        pipe_cls=StableDiffusionUpscalePipeline,
        pipeline=pipeline,
        inputs=inputs,
        inputs_extra=dict(
            image=gooey_gpu.download_images(inputs.image, (512, 512)),
        ),
    )


@app.task(name="diffusion.instruct_pix2pix")
@gooey_gpu.endpoint
def instruct_pix2pix(pipeline: PipelineInfo, inputs: InstructPix2PixInputs):
    return predict_and_upload(
        pipe_cls=StableDiffusionInstructPix2PixPipeline,
        pipeline=pipeline,
        inputs=inputs,
        inputs_extra=dict(
            image=gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE),
        ),
    )


def predict_and_upload(
    *,
    pipe_cls,
    pipeline: PipelineInfo,
    inputs: DiffusersInputs,
    inputs_extra: dict = None,
    extra_components: dict = None,
):
    if inputs_extra is None:
        inputs_extra = {}
    if extra_components is None:
        extra_components = {}
    inputs_dict = inputs.dict()
    inputs_dict.update(inputs_extra)
    # load pipe
    pipe = load_pipe(pipe_cls, pipeline.model_id, pipeline.scheduler, extra_components)
    try:
        # custom safety checker impl
        safety_checker_wrapper(pipe, disabled=pipeline.disable_safety_checker)
        # set seed
        generator = torch.Generator("cuda").manual_seed(pipeline.seed)
        # generate output
        pipe.enable_xformers_memory_efficient_attention()
        with torch.inference_mode():
            output = pipe(**inputs_dict, generator=generator)
    finally:
        # clean up extra components
        for attr in extra_components.keys():
            setattr(pipe, attr, None)
    output_images = output.images
    gooey_gpu.upload_images(output_images, pipeline.upload_urls)


def load_pipe(pipe_cls, model_id: str, scheduler: str, extra_components: dict):
    if issubclass(
        pipe_cls,
        (
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionControlNetPipeline,
        ),
    ):
        base_cls = StableDiffusionPipeline
    else:
        base_cls = pipe_cls
    base_pipe, default_scheduler = _load_pipe_cached(base_cls, model_id)
    base_pipe.scheduler = default_scheduler
    if scheduler:
        base_pipe.scheduler = get_scheduler(base_pipe, scheduler)
    return pipe_cls(**base_pipe.components, **extra_components)


@lru_cache
def _load_pipe_cached(pipe_cls, model_id: str):
    print(f"Loading SD model {model_id!r}...")
    pipe = pipe_cls.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(gooey_gpu.DEVICE_ID)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    return pipe, pipe.scheduler


def get_scheduler(pipe: DiffusionPipeline, cls_name: str):
    for cls in pipe.scheduler.compatibles:
        if cls.__name__ != cls_name:
            continue
        return cls.from_config(pipe.scheduler.config)
    raise ValueError(f"Incompatible scheduler {cls_name!r}")


def safety_checker_wrapper(pipe, disabled: bool):
    def _safety_checker(clip_input, images):
        has_nsfw_concepts = [False] * len(images)
        if not disabled:
            images, has_nsfw_concepts = original(images=images, clip_input=clip_input)
        if any(has_nsfw_concepts):
            raise ValueError(
                "Potential NSFW content was detected in one or more images. "
                "Try again with a different Prompt and/or Regenerate."
            )
        return images, has_nsfw_concepts

    try:
        original = pipe.safety_checker
        if original:
            pipe.safety_checker = _safety_checker
    except AttributeError:
        pass
