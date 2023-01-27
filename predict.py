# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import contextlib
import io
import traceback
import typing
from collections import defaultdict
from time import time

import PIL.Image
import requests
import torch
from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
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
from starlette.responses import Response

app = FastAPI()

MAX_IMAGE_SIZE = (768, 768)


@app.on_event("startup")
def startup() -> None:
    # https://github.com/tiangolo/fastapi/issues/4221
    RunVar("_default_thread_limiter").set(CapacityLimiter(1))


class BaseInputs(BaseModel):
    prompt: typing.List[str]
    negative_prompt: typing.List[str] = None
    num_images_per_prompt: int
    num_inference_steps: int
    guidance_scale: float


class Text2ImgInputs(BaseInputs):
    width: int
    height: int


class Img2ImgInputs(BaseInputs):
    image: typing.List[str]
    strength: float


class InstructPix2PixInputs(BaseInputs):
    image: typing.List[str]
    image_guidance_scale: float


class InpaintInputs(BaseInputs):
    image: typing.List[str]
    mask_image: typing.List[str]


class UpscaleInputs(BaseInputs):
    image: typing.List[str]


class PipelineInfo(BaseModel):
    upload_urls: typing.List[str]
    model_id: str
    scheduler: str = None
    seed: int


@app.post("/text2img/")
def text2img(pipeline: PipelineInfo, inputs: Text2ImgInputs):
    return predictor.endpoint(
        pipeline,
        inputs,
        StableDiffusionPipeline,
    )


@app.post("/img2img/")
def img2img(pipeline: PipelineInfo, inputs: Img2ImgInputs):
    return predictor.endpoint(
        pipeline,
        inputs,
        StableDiffusionImg2ImgPipeline,
        image=download_images(inputs.image),
    )


@app.post("/inpaint/")
def inpaint(pipeline: PipelineInfo, inputs: InpaintInputs):
    return predictor.endpoint(
        pipeline,
        inputs,
        StableDiffusionInpaintPipeline,
        image=download_images(inputs.image),
        mask_image=download_images(inputs.mask_image),
    )


@app.post("/upscale/")
def upscale(pipeline: PipelineInfo, inputs: UpscaleInputs):
    return predictor.endpoint(
        pipeline,
        inputs,
        StableDiffusionUpscalePipeline,
        image=download_images(inputs.image),
    )


@app.post("/instruct_pix2pix/")
def instruct_pix2pix(pipeline: PipelineInfo, inputs: InstructPix2PixInputs):
    return predictor.endpoint(
        pipeline,
        inputs,
        StableDiffusionInstructPix2PixPipeline,
        image=download_images(inputs.image),
    )


PIPES = {
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionInstructPix2PixPipeline,
}


class Predictor:
    pipes = defaultdict(dict)
    schedulers = defaultdict(dict)

    def endpoint(
        self,
        pipeline: PipelineInfo,
        inputs: BaseInputs,
        pipe_cls,
        **kwargs,
    ):
        try:
            self.predict(pipeline, inputs, pipe_cls, **kwargs)
        except Exception as e:
            traceback.print_exc()
            return Response(repr(e), status_code=500)
        else:
            return Response("OK")

    def predict(
        self,
        pipeline: PipelineInfo,
        inputs: BaseInputs,
        pipe_cls,
        **kwargs,
    ):
        print(f" ---> {pipeline!r} {inputs!r}")

        inputs_dict = inputs.dict()
        inputs_dict.update(kwargs)

        pipe = self.load_pipeline(pipeline, pipe_cls)

        s = time()
        with use_in_cuda(pipe):
            generator = torch.Generator("cuda").manual_seed(pipeline.seed)
            output = pipe(**inputs_dict, generator=generator)
            output_images = output.images
        print(f"Prediction time: {time() - s:.3f}s")

        for pil_img, upload_url in zip(output_images, pipeline.upload_urls):
            f = io.BytesIO()
            pil_img.save(f, format="PNG")

            r = requests.put(
                upload_url,
                headers={"Content-Type": "image/png"},
                data=f.getvalue(),
            )
            r.raise_for_status()

    def load_pipeline(
        self,
        pipeline: PipelineInfo,
        pipe_cls,
    ):
        pipes = self.pipes[pipe_cls.__name__]

        try:
            pipe = pipes[pipeline.model_id]
        except KeyError:
            pipe = pipe_cls.from_pretrained(
                pipeline.model_id, torch_dtype=torch.float16
            )
            pipes[pipeline.model_id] = pipe
            self.update_schedulers(pipeline.model_id, pipe)

        try:
            pipe.schduler = self.schedulers[pipeline.model_id][pipeline.scheduler]
        except KeyError:
            raise ValueError(
                f"Incompatible scheduler `{pipeline.scheduler}` for `{pipeline.model_id}`"
            )

        return pipe

    def update_schedulers(self, model_id: str, pipe: DiffusionPipeline):
        schedulers = {None: pipe.scheduler}
        for cls in pipe.scheduler.compatibles:
            try:
                schedulers[cls.__name__] = cls.from_config(pipe.scheduler.config)
            except ImportError as e:
                print(e)
                continue
        self.schedulers[model_id] = schedulers


predictor = Predictor()


def download_images(images: typing.List[str]) -> typing.List[PIL.Image.Image]:
    return [download_image(url) for url in images]


def download_image(url: str) -> PIL.Image.Image:
    bytes = requests.get(url).content
    f = io.BytesIO(bytes)
    image = PIL.Image.open(f).convert("RGB")
    return image


@contextlib.contextmanager
def use_in_cuda(pipe: DiffusionPipeline):
    pipe.to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
        _remove_safety_checker(pipe)
        with torch.inference_mode():
            yield
    finally:
        pipe.to("cpu")
        torch.cuda.empty_cache()


def _remove_safety_checker(pipe):
    # if there's an nsfw filter, replace it with a dummy
    try:
        if pipe.safety_checker:
            pipe.safety_checker = _dummy
    except AttributeError:
        pass


def _dummy(images, **kwargs):
    return images, False
