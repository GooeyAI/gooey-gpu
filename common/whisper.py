import numpy as np
import os
from functools import lru_cache

import requests
import torch
import transformers
from celery.signals import worker_init
from kombu import Queue

import gooey_gpu
from api import PipelineInfo, WhisperInputs, AsrOutput
from celeryconfig import app

QUEUE_PREFIX = os.environ.get("QUEUE_PREFIX", "gooey-gpu")
MODEL_IDS = os.environ["WHISPER_MODEL_IDS"].split()

app.conf.task_queues = app.conf.task_queues or []
for model_id in MODEL_IDS:
    queue = os.path.join(QUEUE_PREFIX, model_id).strip("/")
    app.conf.task_queues.append(Queue(queue))


@worker_init.connect()
def init(**kwargs):
    # app.conf.task_queues = []
    for model_id in MODEL_IDS:
        load_pipe(model_id)


@app.task(name="whisper")
@gooey_gpu.endpoint
def whisper(pipeline: PipelineInfo, inputs: WhisperInputs) -> AsrOutput:
    audio = requests.get(inputs.audio).content
    pipe = load_pipe(pipeline.model_id)

    kwargs = {}
    if inputs.return_timestamps:
        kwargs["return_timestamps"] = True
    if inputs.language:
        kwargs["generate_kwargs"] = dict(
            forced_decoder_ids=pipe.tokenizer.get_decoder_prompt_ids(
                task=inputs.task, language=inputs.language
            )
        )
    # see https://github.com/huggingface/transformers/issues/24707
    old_postprocess = pipe.postprocess
    if inputs.decoder_kwargs:

        def postprocess(model_outputs):
            final_items = []
            key = "tokens"
            for outputs in model_outputs:
                items = outputs[key].numpy()
                final_items.append(items)
            items = np.concatenate(final_items, axis=1)
            items = items.squeeze(0)
            return {"text": pipe.tokenizer.decode(items, **inputs.decoder_kwargs)}

        pipe.postprocess = postprocess

    prediction = pipe(
        audio,
        # see https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor#scrollTo=Ca4YYdtATxzo&line=5&uniqifier=1
        chunk_length_s=inputs.chunk_length_s,
        stride_length_s=inputs.stride_length_s,
        batch_size=inputs.batch_size,
        **kwargs,
    )

    if inputs.decoder_kwargs:
        pipe.postprocess = old_postprocess

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
