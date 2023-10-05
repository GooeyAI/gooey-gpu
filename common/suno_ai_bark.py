import os.path
import tempfile

import bark
import numpy as np
from pydantic import BaseModel

import gooey_gpu
from celeryconfig import app, setup_queues


class PipelineInfo(BaseModel):
    upload_urls: list[str] = []


class BarkInputs(BaseModel):
    prompt: str | list[str]
    history_prompt: str = None
    text_temp: float = 0.7
    waveform_temp: float = 0.7

    init_audio: str = None
    init_transcript: str = None


@app.task(name="bark")
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
            prev_generation, audio_array = bark.generate_audio(
                prompt,
                history_prompt=history_prompt,
                text_temp=inputs.text_temp,
                waveform_temp=inputs.waveform_temp,
                output_full=True,
            )
            audio_chunks.append(audio_array)
    # combine all chunks into long audio file
    audio_array = np.concatenate(audio_chunks)
    # upload the final output
    for url in pipeline.upload_urls:
        gooey_gpu.upload_audio(audio_array, url, rate=bark.SAMPLE_RATE)


setup_queues(
    model_ids=["bark"],
    load_fn=lambda _: bark.preload_models(),
)
