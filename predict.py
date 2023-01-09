# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import contextlib
import typing

import torch
from PIL import Image
from cog import BasePredictor, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DiffusionPipeline,
    SchedulerMixin,
)


class Predictor(BasePredictor):
    sd_pipes = {}
    img2img_pipes = {}

    def predict(
        self,
        hf_model_id: str,
        prompt: str,
        width: int,
        height: int,
        num_outputs: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        negative_prompt: str = None,
        init_image: Path = None,
        strength: float = None,
        scheduler: str = None,
    ) -> typing.List[Path]:
        print(
            " ---> predict("
            f"{hf_model_id=}, {prompt=}, {width=}, {height=}, {num_outputs=}, {num_inference_steps=}, "
            f"{guidance_scale=}, {seed=}, {negative_prompt=}, {init_image=}, {strength=}, {scheduler=}"
            ")"
        )

        if init_image:
            init_image = Image.open(init_image).convert("RGB")

        pipe = self.load_pipeline(hf_model_id=hf_model_id, init_image=init_image)

        with use_scheduler(
            pipe=pipe,
            scheduler=scheduler,
            hf_model_id=hf_model_id,
        ), use_in_cuda(
            pipe=pipe,
        ):
            generator = torch.Generator("cuda").manual_seed(seed)

            if init_image:
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    init_image=init_image,
                    num_images_per_prompt=num_outputs,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                )
            else:
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_outputs,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                )

            output_paths = []
            for i, sample in enumerate(output.images):
                output_path = f"/tmp/out-{i}.png"
                sample.save(output_path)
                output_paths.append(Path(output_path))

            return output_paths

    def load_pipeline(
        self,
        *,
        hf_model_id: str,
        init_image: Path,
    ) -> typing.Union[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline]:
        if init_image:
            cache = self.img2img_pipes
            pipe_cls = StableDiffusionImg2ImgPipeline
        else:
            cache = self.sd_pipes
            pipe_cls = StableDiffusionPipeline

        try:
            pipe = cache[hf_model_id]
        except KeyError:
            pipe = pipe_cls.from_pretrained(hf_model_id, torch_dtype=torch.float16)
            cache[hf_model_id] = pipe

        return pipe


@contextlib.contextmanager
def use_scheduler(
    *,
    pipe: DiffusionPipeline,
    scheduler: str,
    hf_model_id: str,
):
    if not scheduler:
        yield
        return

    default_scheduler = pipe.scheduler
    try:
        pipe.scheduler = get_scheduler_for_pipeline(
            pipe=pipe,
            scheduler=scheduler,
            hf_model_id=hf_model_id,
        )
        yield
    finally:
        pipe.scheduler = default_scheduler


def get_scheduler_for_pipeline(
    *,
    pipe: DiffusionPipeline,
    scheduler: str,
    hf_model_id: str,
) -> SchedulerMixin:

    for cls in pipe.scheduler.compatibles:
        if cls.__name__ == scheduler:
            return cls.from_config(pipe.scheduler.config)

    raise ValueError(f"Incompatible scheduler `{scheduler}` for `{hf_model_id}`")


@contextlib.contextmanager
def use_in_cuda(*, pipe: DiffusionPipeline):
    pipe.to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
        pipe.safety_checker = dummy
        with torch.inference_mode():
            yield
    finally:
        pipe.to("cpu")
        torch.cuda.empty_cache()


def dummy(images, **kwargs):
    return images, False
