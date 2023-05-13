import traceback
from functools import lru_cache

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DiffusionPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from fastapi import APIRouter
from starlette.requests import Request

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
    ControlNetPipelineInfo,
)

app = APIRouter()


@app.post("/text2img/")
@gooey_gpu.endpoint
def text2img(request: Request, pipeline: PipelineInfo, inputs: Text2ImgInputs):
    return predict_and_upload(
        request=request,
        pipe_cls=StableDiffusionPipeline,
        pipeline=pipeline,
        inputs=inputs,
    )


@app.post("/img2img/")
@gooey_gpu.endpoint
def img2img(request: Request, pipeline: PipelineInfo, inputs: Img2ImgInputs):
    return predict_and_upload(
        request=request,
        pipe_cls=StableDiffusionImg2ImgPipeline,
        pipeline=pipeline,
        inputs=inputs,
        inputs_mod=dict(
            image=gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE),
        ),
    )


@app.post("/inpaint/")
@gooey_gpu.endpoint
def inpaint(request: Request, pipeline: PipelineInfo, inputs: InpaintInputs):
    return predict_and_upload(
        request=request,
        pipe_cls=StableDiffusionInpaintPipeline,
        pipeline=pipeline,
        inputs=inputs,
        inputs_mod=dict(
            image=gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE),
            mask_image=gooey_gpu.download_images(inputs.mask_image, MAX_IMAGE_SIZE),
        ),
    )


@app.post("/upscale/")
@gooey_gpu.endpoint
def upscale(request: Request, pipeline: PipelineInfo, inputs: UpscaleInputs):
    return predict_and_upload(
        request=request,
        pipe_cls=StableDiffusionUpscalePipeline,
        pipeline=pipeline,
        inputs=inputs,
        inputs_mod=dict(
            image=gooey_gpu.download_images(inputs.image, (512, 512)),
        ),
    )


@app.post("/instruct_pix2pix/")
@gooey_gpu.endpoint
def instruct_pix2pix(
    request: Request, pipeline: PipelineInfo, inputs: InstructPix2PixInputs
):
    return predict_and_upload(
        request=request,
        pipe_cls=StableDiffusionInstructPix2PixPipeline,
        pipeline=pipeline,
        inputs=inputs,
        inputs_mod=dict(
            image=gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE),
        ),
    )


def predict_and_upload(
    *,
    pipe_cls,
    request: Request,
    pipeline: PipelineInfo,
    inputs: DiffusersInputs,
    inputs_mod: dict = None,
):
    if inputs_mod is None:
        inputs_mod = {}
    inputs_dict = inputs.dict()
    inputs_dict.update(inputs_mod)
    output_images = predict_on_gpu(
        pipeline=pipeline, inputs_dict=inputs_dict, pipe_cls=pipe_cls
    )
    gooey_gpu.upload_images(output_images, pipeline.upload_urls)


@gooey_gpu.gpu_task
def predict_on_gpu(pipeline: PipelineInfo, inputs_dict: dict, pipe_cls):
    # load controlnet
    extra_components = {}
    if isinstance(pipeline, ControlNetPipelineInfo):
        extra_components["controlnet"] = load_controlnet_model(
            pipeline.controlnet_model_id
        )
    # load pipe
    pipe = load_pipe(pipe_cls, pipeline.model_id, extra_components)
    # load scheduler
    pipe.scheduler = get_scheduler(pipeline)
    # gpu inference mode
    with gooey_gpu.use_models(pipe), torch.inference_mode():
        # custom safety checker impl
        safety_checker_wrapper(pipe, disabled=pipeline.disable_safety_checker)
        # set seed
        generator = torch.Generator("cuda").manual_seed(pipeline.seed)
        # generate output
        output = pipe(**inputs_dict, generator=generator)
        output_images = output.images
    return output_images


def load_pipe(pipe_cls, model_id: str, extra_components: dict):
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
    base_pipe = _load_pipe_cached(base_cls, model_id)
    return pipe_cls(**base_pipe.components, **extra_components)


@lru_cache
def _load_pipe_cached(pipe_cls, model_id: str):
    pipe = pipe_cls.from_pretrained(model_id, torch_dtype=torch.float16)
    update_schedulers(model_id, pipe)
    return pipe


@lru_cache
def load_controlnet_model(model_id: str) -> ControlNetModel:
    return ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)


_schedulers = {}


def update_schedulers(model_id: str, pipe: DiffusionPipeline):
    schedulers = {None: pipe.scheduler}
    for cls in pipe.scheduler.compatibles:
        try:
            schedulers[cls.__name__] = cls.from_config(pipe.scheduler.config)
        except ImportError:
            traceback.print_exc()
            continue
    _schedulers[model_id] = schedulers


def get_scheduler(pipeline):
    try:
        return _schedulers[pipeline.model_id][pipeline.scheduler]
    except KeyError:
        raise ValueError(
            f"Incompatible scheduler `{pipeline.scheduler}` for `{pipeline.model_id}`"
        )


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
