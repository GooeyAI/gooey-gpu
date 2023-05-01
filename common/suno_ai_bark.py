import os.path
import tempfile
from functools import lru_cache

import bark
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

import gooey_gpu

app = APIRouter()


class PipelineInfo(BaseModel):
    upload_urls: list[str] = []


class BarkInputs(BaseModel):
    prompt: str | list[str]
    history_prompt: str = None
    text_temp: float = 0.7
    waveform_temp: float = 0.7

    init_audio: str = None
    init_transcript: str = None


@app.post("/bark/")
@gooey_gpu.endpoint
def bark_api(pipeline: PipelineInfo, inputs: BarkInputs):
    assert inputs.prompt, "Please provide a prompt"
    # ensure input is a list
    if isinstance(inputs.prompt, str):
        inputs.prompt = [inputs.prompt]
    history_prompt = inputs.history_prompt
    prev_generation = None
    audio_chunks = []
    # process each input prompt separately
    for prompt in inputs.prompt:
        with tempfile.TemporaryDirectory() as d:
            # save the prev generation as the history prompt
            if prev_generation:
                f = os.path.join(d, "history.npz")
                bark.save_as_prompt(f, prev_generation)
                history_prompt = f
            prev_generation, audio_array = _run_bark(
                prompt=prompt,
                history_prompt=history_prompt,
                inputs=inputs,
            )
            audio_chunks.append(audio_array)
    # combine all chunks into long audio file
    audio_array = np.concatenate(audio_chunks)
    # upload the final output
    for url in pipeline.upload_urls:
        gooey_gpu.upload_audio(audio_array, url, rate=bark.SAMPLE_RATE)


@gooey_gpu.gpu_task
def _run_bark(*, prompt, history_prompt, inputs):
    models = _preload_models()
    with gooey_gpu.use_models(*models):
        return bark.generate_audio(
            prompt,
            history_prompt=history_prompt,
            text_temp=inputs.text_temp,
            waveform_temp=inputs.waveform_temp,
            output_full=True,
        )


@lru_cache
def _preload_models():
    bark.preload_models()
    models = bark.generation.models
    return models["text"]["model"], models["coarse"], models["fine"], models["codec"]
