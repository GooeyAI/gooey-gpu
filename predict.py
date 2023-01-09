# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import contextlib
import typing

import torch
from PIL import Image
from cog import BasePredictor, Path
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


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
        # edit_image: Path,
        # mask_image: Path,
        # upscaling_inference_steps: int,
        # scheduler: str = Input(
        #     default="DPMSolverMultistep",
        #     choices=[
        #         "DDIM",
        #         "K_EULER",
        #         "DPMSolverMultistep",
        #         "K_EULER_ANCESTRAL",
        #         "PNDM",
        #         "KLMS",
        #     ],
        #     description="Choose a scheduler.",
        # ),
    ) -> typing.List[Path]:

        if init_image:
            init_image = Image.open(init_image).convert("RGB")
            store = self.img2img_pipes
            pipe_cls = StableDiffusionImg2ImgPipeline
        else:
            store = self.sd_pipes
            pipe_cls = StableDiffusionPipeline

        try:
            pipe = store[hf_model_id]
        except KeyError:
            pipe = pipe_cls.from_pretrained(hf_model_id, torch_dtype=torch.float16)
            store[hf_model_id] = pipe

        with use_in_cuda(pipe):
            pipe.enable_xformers_memory_efficient_attention()
            pipe.safety_checker = dummy

            generator = torch.Generator("cuda").manual_seed(seed)

            if init_image:
                output = pipe(
                    prompt=[prompt] * num_outputs,
                    negative_prompt=[negative_prompt] * num_outputs
                    if negative_prompt
                    else None,
                    init_image=[init_image] * num_outputs if init_image else None,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                )
            else:
                output = pipe(
                    prompt=[prompt] * num_outputs,
                    negative_prompt=[negative_prompt] * num_outputs
                    if negative_prompt
                    else None,
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


def dummy(images, **kwargs):
    return images, False


@contextlib.contextmanager
def use_in_cuda(pipe):
    pipe.to("cuda")
    try:
        with torch.inference_mode():
            yield
    finally:
        pipe.to("cpu")
        torch.cuda.empty_cache()
