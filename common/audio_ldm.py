import os
from functools import lru_cache

import torch
from diffusers import AudioLDMPipeline

import gooey_gpu
from api import PipelineInfo, AudioLDMInputs
from celeryconfig import app, setup_queues


@app.task(name="audio_ldm")
@gooey_gpu.endpoint
def audio_ldm(pipeline: PipelineInfo, inputs: AudioLDMInputs):
    pipe = load_model(pipeline.model_id)
    with gooey_gpu.use_models(pipe), torch.inference_mode():
        audios = pipe(**inputs.dict()).audios
    gooey_gpu.apply_parallel(gooey_gpu.upload_audio, audios, pipeline.upload_urls)


@lru_cache
def load_model(model_id):
    return AudioLDMPipeline.from_pretrained(model_id)


setup_queues(
    model_ids=os.environ["AUDIO_LDM_MODEL_IDS"].split(),
    load_fn=load_model,
)
