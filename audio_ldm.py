import io
from functools import lru_cache

import numpy as np
import requests
import scipy
import torch
from diffusers import AudioLDMPipeline
from fastapi import APIRouter

import gooey_gpu
from models import PipelineInfo, AudioLDMInputs

app = APIRouter()


@app.post("/audio_ldm/")
@gooey_gpu.endpoint
def audio_ldm(pipeline: PipelineInfo, inputs: AudioLDMInputs):
    # generate audio on gpu
    audios = run_audio_ldm(pipeline, inputs)
    # upload audios
    gooey_gpu.apply_parallel(upload_audio, audios, pipeline.upload_urls)


@gooey_gpu.gpu_task
def run_audio_ldm(pipeline: PipelineInfo, inputs: AudioLDMInputs) -> io.BytesIO:
    # Inspired by Stable Diffusion, AudioLDM
    # is a text-to-audio latent diffusion model (LDM) that learns continuous audio representations from CLAP
    # latents. AudioLDM takes a text prompt as input and predicts the corresponding audio. It can generate text-conditional
    # sound effects, human speech and music.
    pipe = get_audio_ldm_pipeline(pipeline.model_id)
    with gooey_gpu.use_models(pipe), torch.inference_mode():
        return pipe(**inputs.dict()).audios


@lru_cache
def get_audio_ldm_pipeline(model_id):
    return AudioLDMPipeline.from_pretrained(model_id)


def upload_audio(audio: np.ndarray, url: str):
    # The resulting audio output can be saved as a .wav file:
    f = io.BytesIO()
    scipy.io.wavfile.write(f, rate=16000, data=audio)
    audio_bytes = f.getvalue()
    # upload to given url
    r = requests.put(url, headers={"Content-Type": "audio/wav"}, data=audio_bytes)
    r.raise_for_status()
