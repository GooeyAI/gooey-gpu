import os
from functools import lru_cache

import numpy as np
import torch
import transformers

import gooey_gpu
from api import PipelineInfo, TTSInputs
from celeryconfig import app, setup_queues



@app.task(name="tts")
@gooey_gpu.endpoint
def tts(pipeline: PipelineInfo, inputs: TTSInputs) -> None:
    assert inputs.text, "Please provide text to convert to speech"

    pipe = load_pipe(pipeline.model_id)

    forward_params = {}
    if inputs.language is not None:
        forward_params["language"] = inputs.language

    with torch.no_grad():
        result = pipe(inputs.text, forward_params=forward_params)

    # Normalize output across transformers versions
    if isinstance(result, dict):
        audio_obj = result.get("audio", result)
        if isinstance(audio_obj, dict) and "array" in audio_obj:
            audio_array = audio_obj["array"]
            sampling_rate = audio_obj.get("sampling_rate")
        else:
            audio_array = np.asarray(audio_obj)
            sampling_rate = result.get("sampling_rate")
    else:
        audio_array = np.asarray(result)
        sampling_rate = None

    if sampling_rate is None:
        try:
            sampling_rate = pipe.model.config.sampling_rate
        except Exception:
            sampling_rate = 22050

    for url in pipeline.upload_urls:
        gooey_gpu.upload_audio(audio_array, url, rate=sampling_rate)


@lru_cache
def load_pipe(model_id: str):
    print(f"Loading TTS model {model_id!r}...")
    pipe = transformers.pipeline(
        "text-to-speech",
        model=model_id,
        device=gooey_gpu.DEVICE_ID,
        torch_dtype=torch.float16,
    )
    return pipe


setup_queues(
    model_ids=os.environ["TTS_MODEL_IDS"].split(),
    load_fn=load_pipe,
)
