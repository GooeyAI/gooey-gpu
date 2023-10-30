import os
from functools import lru_cache

import requests
import torch
import transformers

import gooey_gpu
from api import SeamlessM4TInputs, SeamlessM4TPipeline, AsrOutput
from celeryconfig import app, setup_queues


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

    prediction = pipe(
        audio,
        # see https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor#scrollTo=Ca4YYdtATxzo&line=5&uniqifier=1
        chunk_length_s=inputs.chunk_length_s,
        stride_length_s=inputs.stride_length_s,
        batch_size=inputs.batch_size,
        generate_kwargs=dict(
            tgt_lang=inputs.tgt_lang,
        ),
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
