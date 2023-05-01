from functools import lru_cache

import torch
from diffusers import (
    IFPipeline,
    StableDiffusionUpscalePipeline,
)
from fastapi import APIRouter
from pydantic import BaseModel
from transformers import T5EncoderModel

import gooey_gpu
from common.diffusion import safety_checker_wrapper
from common.pipeline_if_sr_patch import IFSuperResolutionPipelinePatch

app = APIRouter(prefix="/deepfloyd_if")


class PipelineInfo(BaseModel):
    upload_urls: list[str] = []
    model_id: tuple[str, str, str]
    seed: int = 42
    disable_safety_checker: bool = False


class DeepfloydInputs(BaseModel):
    prompt: list[str]
    negative_prompt: list[str] = None
    num_inference_steps: tuple[int, int, int] = (100, 50, 75)
    num_images_per_prompt: int = 1
    guidance_scale: tuple[float, float, float] = (7, 4, 9)


class Text2ImgInputs(DeepfloydInputs):
    width: int
    height: int


@app.post("/text2img/")
@gooey_gpu.endpoint
def text2img(pipeline: PipelineInfo, inputs: Text2ImgInputs):
    output_images = _run_model(pipeline, inputs)
    gooey_gpu.upload_images(output_images, pipeline.upload_urls)


@gooey_gpu.gpu_task
def _run_model(pipeline: PipelineInfo, inputs: Text2ImgInputs):
    pipe1 = load_pipe1(pipeline.model_id[0])
    pipe2 = load_pipe2(pipeline.model_id[1])
    pipe3 = load_pipe3(pipeline.model_id[2])

    inputs.prompt *= inputs.num_images_per_prompt
    if inputs.negative_prompt:
        inputs.negative_prompt *= inputs.num_images_per_prompt

    with gooey_gpu.use_models(pipe1), torch.inference_mode():
        generator = torch.Generator().manual_seed(pipeline.seed)
        # custom safety checker impl
        safety_checker_wrapper(pipe1, disabled=pipeline.disable_safety_checker)
        # Create text embeddings
        prompt_embeds, negative_embeds = pipe1.encode_prompt(
            inputs.prompt, negative_prompt=inputs.negative_prompt
        )
        # The main diffusion process
        images = pipe1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            guidance_scale=inputs.guidance_scale[0],
            num_inference_steps=inputs.num_inference_steps[0],
            output_type="pt",
            generator=generator,
            width=inputs.width // 16,
            height=inputs.height // 16,
        ).images

    with gooey_gpu.use_models(pipe2), torch.inference_mode():
        # custom safety checker impl
        safety_checker_wrapper(pipe2, disabled=pipeline.disable_safety_checker)
        # Super Resolution 64x64 to 256x256
        images = pipe2(
            image=images,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            guidance_scale=inputs.guidance_scale[1],
            num_inference_steps=inputs.num_inference_steps[1],
            output_type="pt",
            generator=generator,
        ).images

    with gooey_gpu.use_models(pipe3), torch.inference_mode():
        # custom safety checker impl
        safety_checker_wrapper(pipe3, disabled=pipeline.disable_safety_checker)
        # Super Resolution 256x256 to 1024x1024
        output_images = pipe3(
            image=images,
            prompt=inputs.prompt,
            negative_prompt=inputs.negative_prompt,
            guidance_scale=inputs.guidance_scale[2],
            num_inference_steps=inputs.num_inference_steps[2],
            generator=generator,
        ).images

    return output_images


@lru_cache
def load_pipe1(model_id: str):
    # text_encoder = T5EncoderModel.from_pretrained(
    #     model_id,
    #     subfolder="text_encoder",
    #     load_in_8bit=True,
    #     variant="8bit",
    # )
    return IFPipeline.from_pretrained(
        model_id,
        # text_encoder=text_encoder,
        variant="fp16",
        torch_dtype=torch.float16,
    )


@lru_cache
def load_pipe2(model_id: str):
    return IFSuperResolutionPipelinePatch.from_pretrained(
        model_id,
        text_encoder=None,  # no use of text encoder => memory savings!
        variant="fp16",
        torch_dtype=torch.float16,
    )


@lru_cache
def load_pipe3(model_id: str):
    return StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
