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
from lavis.models import load_model_and_preprocess
from pydantic import BaseModel

import deforum_script
import gooey_gpu

app = FastAPI()
gooey_gpu.register_app(app)

MAX_IMAGE_SIZE = (768, 768)


class PipelineInfo(BaseModel):
    upload_urls: typing.List[str] = []
    model_id: str
    scheduler: str = None
    seed: int = 42


class VQAInput(BaseModel):
    image: typing.List[str]
    question: typing.List[str]

    # https://github.com/salesforce/LAVIS/blob/7aa83e93003dade66f7f7eaba253b10c459b012d/lavis/models/blip_models/blip_vqa.py#L162
    num_beams: int = 3
    inference_method: str = "generate"
    max_len: int = 10
    min_len: int = 1
    num_ans_candidates: int = 128


@app.post("/vqa/")
@gooey_gpu.endpoint
def vqa(pipeline: PipelineInfo, inputs: VQAInput):
    # load model
    model_id = pipeline.model_id.split("/")
    model, vis_processors, txt_processors = _load_lavis_model(*model_id)
    # get inputs
    inputs_kwargs = inputs.dict()
    image = gooey_gpu.download_images(inputs_kwargs.pop("image"), MAX_IMAGE_SIZE)
    question = inputs_kwargs.pop("question")
    # do inference
    with gooey_gpu.use_models(model):
        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = torch.stack([vis_processors["eval"](im) for im in image]).to(
            gooey_gpu.DEVICE_ID
        )
        question = [txt_processors["eval"](q) for q in question]
        # generate answerss
        return model.predict_answers(
            samples={"image": image, "text_input": question}, **inputs_kwargs
        )
        # ['singapore']


class ImageCaptioningInput(BaseModel):
    image: typing.List[str]

    # https://github.com/salesforce/LAVIS/blob/7aa83e93003dade66f7f7eaba253b10c459b012d/lavis/models/blip_models/blip_caption.py#L136
    num_beams = 3
    max_length = 30
    min_length = 10
    repetition_penalty = 1.0
    num_captions = 1


@app.post("/image-captioning/")
@gooey_gpu.endpoint
def image_captioning(pipeline: PipelineInfo, inputs: ImageCaptioningInput):
    # load model
    model_id = pipeline.model_id.split("/")
    model, vis_processors, txt_processors = _load_lavis_model(*model_id)
    # get inputs
    inputs_kwargs = inputs.dict()
    image = gooey_gpu.download_images(inputs_kwargs.pop("image"), MAX_IMAGE_SIZE)
    # do inference
    with gooey_gpu.use_models(model):
        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = torch.stack([vis_processors["eval"](im) for im in image]).to(
            gooey_gpu.DEVICE_ID
        )
        # generate caption
        return model.generate(samples={"image": image}, **inputs_kwargs)
        # ['a large fountain spewing water into the air']


_lavis_cache = {}


def _load_lavis_model(name, model_type):
    try:
        ret = _lavis_cache[(name, model_type)]
    except KeyError:
        ret = load_model_and_preprocess(name, model_type, is_eval=True)
        _lavis_cache[(name, model_type)] = ret
    return ret


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
        _replace_safety_checker(pipe)
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


def _replace_safety_checker(pipe):
    def _safety_checker(images, clip_input):
        # images, has_nsfw_concepts = original(images=images, clip_input=clip_input)
        # if has_nsfw_concepts:
        #     raise ValueError(
        #         "Potential NSFW content was detected in one or more images. "
        #         "Try again with a different Prompt and/or Regenerate."
        #     )
        return images, False

    try:
        original = pipe.safety_checker
        if original:
            pipe.safety_checker = _safety_checker
    except AttributeError:
        pass
