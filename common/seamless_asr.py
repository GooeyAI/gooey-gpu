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


class SeamlessM4TPipeline(BaseModel):
    upload_urls: typing.List[str] = []
    model_id: typing.Literal[
        "facebook/hf-seamless-m4t-large", "facebook/hf-seamless-m4t-medium"
    ] = "facebook/hf-seamless-m4t-large"


class SeamlessM4TInputs(BaseModel):
    audio: str | None  # required for ASR, S2ST, and S2TT
    text: str | None  # required for T2ST and T2TT
    task: typing.Literal["S2ST", "T2ST", "S2TT", "T2TT", "ASR"] = "ASR"
    src_lang: str | None = None  # required for T2ST and T2TT
    tgt_lang: str | None = None  # ignored for ASR (only src_lang is used)
    # seamless uses ISO 639-3 codes for languages

    chunk_length_s: float = 30
    stride_length_s: typing.Tuple[float, float] = (6, 0)
    batch_size: int = 16


class SeamlessM4TOutput(typing.TypedDict):
    text: str | None
    audio: str | None


@app.task(name="seamless")
@gooey_gpu.endpoint
def seamless_asr(
    pipeline: SeamlessM4TPipeline,
    inputs: SeamlessM4TInputs,
) -> AsrOutput:
    audio = requests.get(inputs.audio).content
    pipe = load_pipe(pipeline.model_id)

    previous_src_lang = None
    if inputs.src_lang:
        previous_src_lang = pipe.tokenizer.src_lang
        pipe.tokenizer.src_lang = inputs.src_lang

    tgt_lang = inputs.tgt_lang or inputs.src_lang or "eng"

    prediction = pipe(
        audio,
        # see https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor#scrollTo=Ca4YYdtATxzo&line=5&uniqifier=1
        chunk_length_s=inputs.chunk_length_s,
        stride_length_s=inputs.stride_length_s,
        batch_size=inputs.batch_size,
        generate_kwargs=dict(tgt_lang=tgt_lang),
    )

    if previous_src_lang:
        pipe.tokenizer.src_lang = previous_src_lang

    return prediction


@lru_cache
def load_pipe(model_id: str):
    print(f"Loading asr model {model_id!r}...")
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=gooey_gpu.DEVICE_ID,
        torch_dtype=torch.float16,
    )
    return pipe


setup_queues(
    model_ids=os.environ["SEAMLESS_MODEL_IDS"].split(),
    load_fn=load_pipe,
)
