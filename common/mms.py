import os
from functools import lru_cache

import requests
import transformers

import gooey_gpu
from api import PipelineInfo, WhisperInputs, AsrOutput
from celeryconfig import app, setup_queues


@app.task(name="mms")
@gooey_gpu.endpoint
def mms(pipeline: PipelineInfo, inputs: WhisperInputs) -> AsrOutput:
    audio = requests.get(inputs.audio).content
    pipe = load_pipe(pipeline.model_id)

    pipe.tokenizer.set_target_lang(inputs.language)
    pipe.model.load_adapter(inputs.language)

    kwargs = {}
    if inputs.return_timestamps:
        kwargs["return_timestamps"] = "word"

    prediction = pipe(
        audio,
        # see https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor#scrollTo=Ca4YYdtATxzo&line=5&uniqifier=1
        chunk_length_s=inputs.chunk_length_s,
        stride_length_s=inputs.stride_length_s,
        batch_size=inputs.batch_size,
        **kwargs,
    )

    return prediction


@lru_cache
def load_pipe(model_id: str):
    print(f"Loading asr model {model_id!r}...")
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=gooey_gpu.DEVICE_ID,
    )
    return pipe


setup_queues(
    model_ids=os.environ["MMS_MODEL_IDS"].split(),
    load_fn=load_pipe,
)
