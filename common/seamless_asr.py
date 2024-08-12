import os
import typing
from functools import lru_cache

import requests
import torch
import transformers
from pydantic import BaseModel

import gooey_gpu
from api import AsrOutput
from celeryconfig import app, setup_queues


class SeamlessASRPipeline(BaseModel):
    model_id: str


class SeamlessASRInputs(BaseModel):
    audio: str
    src_lang: str
    tgt_lang: str | None = None

    chunk_length_s: float = 30
    stride_length_s: typing.Tuple[float, float] = (6, 0)
    batch_size: int = 16


@app.task(name="seamless.asr")
@gooey_gpu.endpoint
def seamless_asr(
    pipeline: SeamlessASRPipeline,
    inputs: SeamlessASRInputs,
) -> AsrOutput:
    audio = requests.get(inputs.audio).content
    pipe = load_pipe(pipeline.model_id)

    previous_src_lang = pipe.tokenizer.src_lang
    if inputs.src_lang:
        pipe.tokenizer.src_lang = inputs.src_lang

    tgt_lang = inputs.tgt_lang or inputs.src_lang

    prediction = pipe(
        audio,
        # see https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor#scrollTo=Ca4YYdtATxzo&line=5&uniqifier=1
        chunk_length_s=inputs.chunk_length_s,
        stride_length_s=inputs.stride_length_s,
        batch_size=inputs.batch_size,
        generate_kwargs=dict(tgt_lang=tgt_lang),
    )

    pipe.tokenizer.src_lang = previous_src_lang

    return prediction


@lru_cache
def load_pipe(model_id: str) -> transformers.AutomaticSpeechRecognitionPipeline:
    print(f"Loading seamless m4t pipeline {model_id!r}...")
    pipe = typing.cast(
        transformers.AutomaticSpeechRecognitionPipeline,
        transformers.pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=gooey_gpu.DEVICE_ID,
            torch_dtype=torch.float16,
        ),
    )
    return pipe


setup_queues(
    model_ids=os.environ["SEAMLESS_MODEL_IDS"].split(),
    load_fn=load_pipe,
)
