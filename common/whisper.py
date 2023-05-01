from functools import lru_cache

import requests
import torch
import transformers
from fastapi import APIRouter

import gooey_gpu
from api import PipelineInfo, WhisperInputs, AsrOutput

app = APIRouter()


@app.post("/whisper/")
@gooey_gpu.endpoint
def whisper(pipeline: PipelineInfo, inputs: WhisperInputs) -> AsrOutput:
    audio = requests.get(inputs.audio).content
    prediction = run_whisper(audio, inputs, pipeline.model_id)
    return prediction


@gooey_gpu.gpu_task
def run_whisper(audio: bytes, inputs: WhisperInputs, model_id: str):
    pipe = load_pipe(model_id)
    with gooey_gpu.use_models(
        pipe.model, pipe.model.get_encoder(), pipe.model.get_decoder()
    ):
        pipe.device = torch.device(gooey_gpu.DEVICE_ID)
        generate_kwargs = {}
        if inputs.language:
            generate_kwargs[
                "forced_decoder_ids"
            ] = pipe.tokenizer.get_decoder_prompt_ids(
                task=inputs.task, language=inputs.language
            )
        prediction = pipe(
            audio,
            return_timestamps=inputs.return_timestamps,
            generate_kwargs=generate_kwargs,
            # see https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor#scrollTo=Ca4YYdtATxzo&line=5&uniqifier=1
            chunk_length_s=30,
            stride_length_s=[6, 0],
            batch_size=128,
        )
    return prediction


@lru_cache
def load_pipe(model_id: str):
    pipe = transformers.pipeline("automatic-speech-recognition", model=model_id)
    return pipe
