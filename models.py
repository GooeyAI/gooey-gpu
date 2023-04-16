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
    num_images_per_prompt: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


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


class AudioLDMInputs(BaseModel):
    prompt: typing.List[str]
    negative_prompt: typing.List[str] = None
    num_waveforms_per_prompt: int = 1
    num_inference_steps: int = 10
    guidance_scale: float = 2.5
    audio_length_in_s: float = 5.12


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


class WhisperInputs(BaseModel):
    audio: str
    task: typing.Literal["translate", "transcribe"] = "transcribe"
    language: str = None
    return_timestamps: bool = False


class NemoASRInputs(BaseModel):
    audio: str


class AsrOutputChunk(BaseModel):
    timestamp: tuple[float, float]
    text: str


class AsrOutput(BaseModel):
    text: str
    chunks: list[AsrOutputChunk] = []
