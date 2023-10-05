import os
from functools import lru_cache

import numpy as np
import requests
import torch
import transformers

import gooey_gpu
from api import PipelineInfo, WhisperInputs, AsrOutput
from celeryconfig import app, setup_queues


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


setup_queues(
    model_ids=os.environ["WHISPER_MODEL_IDS"].split(),
    load_fn=load_pipe,
)
