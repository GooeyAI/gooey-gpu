# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import contextlib
import typing
from collections import defaultdict

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
    schedulers = defaultdict(dict)

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

        pipe = self.load_pipeline(
            hf_model_id=hf_model_id,
            scheduler=scheduler,
            init_image=init_image,
        )

        with use_in_cuda(pipe):
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
        scheduler: str,
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
            self.update_schedulers(hf_model_id, pipe)

        try:
            pipe.schduler = self.schedulers[hf_model_id][scheduler]
        except KeyError:
            raise ValueError(
                f"Incompatible scheduler `{scheduler}` for `{hf_model_id}`"
            )

        return pipe

    def update_schedulers(self, hf_model_id: str, pipe: DiffusionPipeline):
        schedulers = {None: pipe.scheduler}
        for cls in pipe.scheduler.compatibles:
            try:
                schedulers[cls.__name__] = cls.from_config(pipe.scheduler.config)
            except ImportError as e:
                print(e)
                continue
        self.schedulers[hf_model_id] = schedulers


@contextlib.contextmanager
def use_in_cuda(pipe: DiffusionPipeline):
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
