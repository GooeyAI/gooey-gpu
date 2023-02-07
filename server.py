import shutil
import traceback
import typing
import uuid
from collections import defaultdict

import requests
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

import deforum_script
import gooey_gpu

app = FastAPI()
gooey_gpu.register_app(app)

MAX_IMAGE_SIZE = (768, 768)


class PipelineInfo(BaseModel):
    upload_urls: typing.List[str]
    model_id: str
    scheduler: str = None
    seed: int


@app.post("/deforum/")
@gooey_gpu.endpoint
def deforum(pipeline: PipelineInfo, inputs: deforum_script.DeforumAnimArgs):
    # init args
    args = deforum_script.DeforumArgs(batch_name=str(uuid.uuid1()))
    args.seed = pipeline.seed
    if pipeline.scheduler:
        args.sampler = pipeline.scheduler
    anim_args = deforum_script.DeforumAnimArgs()
    for k, v in inputs.dict().items():
        setattr(anim_args, k, v)
    try:
        # run inference
        args, anim_args = gooey_gpu.run_in_gpu(
            app=app,
            fn=_deforum,
            kwargs=dict(pipeline=pipeline, args=args, anim_args=anim_args),
        )
        # generate video
        vid_path = deforum_script.create_video(args, anim_args)
        with open(vid_path, "rb") as f:
            vid_bytes = f.read()
    finally:
        # cleanup
        shutil.rmtree(args.outdir, ignore_errors=True)
    # upload videos
    for url in pipeline.upload_urls:
        r = requests.put(
            url,
            headers={"Content-Type": "video/mp4"},
            data=vid_bytes,
        )
        r.raise_for_status()
        return


def _deforum(pipeline: PipelineInfo, args, anim_args):
    root = _load_deforum(pipeline)
    with gooey_gpu.use_models(root.model):
        deforum_script.run(root, args, anim_args)
    return args, anim_args


_deforum_cache = {}


def _load_deforum(pipeline):
    try:
        root = _deforum_cache[pipeline.model_id]
    except KeyError:
        root = deforum_script.Root()
        root.map_location = gooey_gpu.DEVICE_ID
        root.model_checkpoint = pipeline.model_id
        deforum_script.setup(root)
        _deforum_cache[pipeline.model_id] = root
    return root


def obj_to_dict(obj):
    return {k: v for k, v in vars(obj).items() if not k.startswith("__")}


class DiffusersInputs(BaseModel):
    prompt: typing.List[str]
    negative_prompt: typing.List[str] = None
    num_images_per_prompt: int
    num_inference_steps: int
    guidance_scale: float


class Text2ImgInputs(DiffusersInputs):
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


class Img2ImgInputs(DiffusersInputs):
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


class InpaintInputs(DiffusersInputs):
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


class UpscaleInputs(DiffusersInputs):
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


class InstructPix2PixInputs(DiffusersInputs):
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


def predict(pipeline: PipelineInfo, inputs: DiffusersInputs, pipe_cls, **kwargs):
    inputs_dict = inputs.dict()
    inputs_dict.update(kwargs)
    output_images = gooey_gpu.run_in_gpu(
        app=app,
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
        _remove_safety_checker(pipe)
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

    try:
        pipe = pipes[pipeline.model_id]
    except KeyError:
        pipe = pipe_cls.from_pretrained(pipeline.model_id, torch_dtype=torch.float16)
        pipes[pipeline.model_id] = pipe
        update_schedulers(pipeline.model_id, pipe)

    try:
        pipe.schduler = _schedulers_cache[pipeline.model_id][pipeline.scheduler]
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
    _schedulers_cache[model_id] = schedulers


def _remove_safety_checker(pipe):
    """If there's an nsfw filter, replace it with a dummy"""
    try:
        if pipe.safety_checker:
            pipe.safety_checker = _dummy
    except AttributeError:
        pass


def _dummy(images, **kwargs):
    return images, False
