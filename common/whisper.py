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
    generate_kwargs = {}
    if inputs.language:
        generate_kwargs["forced_decoder_ids"] = pipe.tokenizer.get_decoder_prompt_ids(
            task=inputs.task, language=inputs.language
        )
    prediction = pipe(
        audio,
        return_timestamps=inputs.return_timestamps,
        generate_kwargs=generate_kwargs,
        # see https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor#scrollTo=Ca4YYdtATxzo&line=5&uniqifier=1
        chunk_length_s=30,
        stride_length_s=[6, 0],
        batch_size=16,
    )
    return prediction


@lru_cache
def load_pipe(model_id: str):
    print(f"Loading whisper model {model_id!r}...")
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=gooey_gpu.DEVICE_ID,
        torch_dtype=torch.float16,
    )
    return pipe
