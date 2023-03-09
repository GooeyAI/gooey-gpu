import typing

from pydantic import BaseModel

MAX_IMAGE_SIZE = (768, 768)


class PipelineInfo(BaseModel):
    upload_urls: typing.List[str] = []
    model_id: str
    scheduler: str = None
    seed: int = 42
    disable_safety_checker: bool = False


class DiffusersInputs(BaseModel):
    prompt: typing.List[str]
    negative_prompt: typing.List[str] = None
    num_images_per_prompt: int
    num_inference_steps: int
    guidance_scale: float


class Text2ImgInputs(DiffusersInputs):
    width: int
    height: int


class Img2ImgInputs(DiffusersInputs):
    image: typing.List[str]
    strength: float


class InpaintInputs(DiffusersInputs):
    image: typing.List[str]
    mask_image: typing.List[str]


class UpscaleInputs(DiffusersInputs):
    image: typing.List[str]


class InstructPix2PixInputs(DiffusersInputs):
    image: typing.List[str]
    image_guidance_scale: float


class ControlNetPipelineInfo(PipelineInfo):
    controlnet_model_id: str
    disable_preprocessing: bool = False


class ControlNetInputs(DiffusersInputs):
    image: typing.List[str]


class VQAInput(BaseModel):
    image: typing.List[str]
    question: typing.List[str]

    # https://github.com/salesforce/LAVIS/blob/7aa83e93003dade66f7f7eaba253b10c459b012d/lavis/models/blip_models/blip_vqa.py#L162
    num_beams: int = 3
    inference_method: str = "generate"
    max_len: int = 10
    min_len: int = 1
    num_ans_candidates: int = 128


class ImageCaptioningInput(BaseModel):
    image: typing.List[str]

    # https://github.com/salesforce/LAVIS/blob/7aa83e93003dade66f7f7eaba253b10c459b012d/lavis/models/blip_models/blip_caption.py#L136
    num_beams = 3
    max_length = 30
    min_length = 10
    repetition_penalty = 1.0
    num_captions = 1
