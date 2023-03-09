import traceback
from collections import defaultdict

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DiffusionPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline,
    ControlNetModel,
)
from fastapi import APIRouter
from starlette.requests import Request

import gooey_gpu
from models import (
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
    return predict(
        request=request,
        pipe_cls=StableDiffusionPipeline,
        pipeline=pipeline,
        inputs=inputs,
    )


@app.post("/img2img/")
@gooey_gpu.endpoint
def img2img(request: Request, pipeline: PipelineInfo, inputs: Img2ImgInputs):
    return predict(
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
    return predict(
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
    return predict(
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
    return predict(
        request=request,
        pipe_cls=StableDiffusionInstructPix2PixPipeline,
        pipeline=pipeline,
        inputs=inputs,
        inputs_mod=dict(
            image=gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE),
        ),
    )


def predict(
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
    output_images = gooey_gpu.run_in_gpu(
        app=request.app,
        fn=_predict,
        kwargs=dict(
            pipeline=pipeline,
            inputs_dict=inputs_dict,
            pipe_cls=pipe_cls,
        ),
    )
    gooey_gpu.upload_images(output_images, pipeline.upload_urls)


def _predict(pipeline: PipelineInfo, inputs_dict: dict, pipe_cls):
    pipe = load_pipeline(pipeline, pipe_cls)

    with gooey_gpu.use_models(pipe), torch.inference_mode():
        safety_checker_wrapper(pipe, disabled=pipeline.disable_safety_checker)
        pipe.enable_xformers_memory_efficient_attention()

        generator = torch.Generator("cuda").manual_seed(pipeline.seed)
        output = pipe(**inputs_dict, generator=generator)

        output_images = output.images

    return output_images


_pipes_cache = defaultdict(dict)
_schedulers_cache = defaultdict(dict)


def load_pipeline(
    pipeline: PipelineInfo,
    pipe_cls,
):
    pipes = _pipes_cache[pipe_cls.__name__]

    if isinstance(pipeline, ControlNetPipelineInfo):
        controlnet = _load_controlnet_model(pipeline.controlnet_model_id)
    else:
        controlnet = None

    try:
        pipe = pipes[pipeline.model_id]
        pipe.controlnet = controlnet
    except KeyError:
        if controlnet:
            pipe = pipe_cls.from_pretrained(
                pipeline.model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16,
            )
        else:
            pipe = pipe_cls.from_pretrained(
                pipeline.model_id,
                torch_dtype=torch.float16,
            )
        pipes[pipeline.model_id] = pipe
        update_schedulers(pipeline.model_id, pipe)

    try:
        pipe.scheduler = _schedulers_cache[pipeline.model_id][pipeline.scheduler]
    except KeyError:
        raise ValueError(
            f"Incompatible scheduler `{pipeline.scheduler}` for `{pipeline.model_id}`"
        )

    return pipe


_controlnet_cache = {}


def _load_controlnet_model(model_id: str) -> ControlNetModel:
    try:
        model = _controlnet_cache[model_id]
    except KeyError:
        model = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
        _controlnet_cache[model_id] = model
    return model


def update_schedulers(model_id: str, pipe: DiffusionPipeline):
    schedulers = {None: pipe.scheduler}
    for cls in pipe.scheduler.compatibles:
        try:
            schedulers[cls.__name__] = cls.from_config(pipe.scheduler.config)
        except ImportError:
            traceback.print_exc()
            continue
    _schedulers_cache[model_id] = schedulers


def safety_checker_wrapper(pipe, disabled: bool):
    def _safety_checker(images, clip_input):
        if disabled:
            return images, False
        images, has_nsfw_concepts = original(images=images, clip_input=clip_input)
        if has_nsfw_concepts:
            raise ValueError(
                "Potential NSFW content was detected in one or more images. "
                "Try again with a different Prompt and/or Regenerate."
            )

    try:
        original = pipe.safety_checker
        if original:
            pipe.safety_checker = _safety_checker
    except AttributeError:
        pass
