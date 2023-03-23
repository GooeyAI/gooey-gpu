from functools import lru_cache

import requests
import torch
import transformers
from fastapi import APIRouter
from pydantic import BaseModel

import gooey_gpu
from models import PipelineInfo

app = APIRouter()


class WhisperInputs(BaseModel):
    audio: str


@app.post("/whisper/")
@gooey_gpu.endpoint
def whisper(pipeline: PipelineInfo, inputs: WhisperInputs) -> list[dict]:
    audio = requests.get(inputs.audio).content
    prediction = run_whisper(audio, pipeline.model_id)
    return prediction


@gooey_gpu.gpu_task
def run_whisper(audio: bytes, model_id: str):
    pipe = load_pipe(model_id)
    with gooey_gpu.use_models(
        pipe.model, pipe.model.get_encoder(), pipe.model.get_decoder()
    ):
        pipe.device = torch.device(gooey_gpu.DEVICE_ID)
        prediction = pipe(
            audio,
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=[6, 0],
            batch_size=32,
        )["chunks"]
    return prediction


@lru_cache
def load_pipe(model_id: str):
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=30,
    )
    return pipe
