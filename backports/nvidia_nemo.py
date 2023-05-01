import os.path
import tempfile
from functools import lru_cache

import nemo.collections.asr as nemo_asr
import requests
from fastapi import APIRouter

import gooey_gpu
from api import PipelineInfo, AsrOutput, NemoASRInputs

app = APIRouter()


@app.post("/nemo/asr/")
@gooey_gpu.endpoint
def nemo_asr_api(pipeline: PipelineInfo, inputs: NemoASRInputs) -> AsrOutput:
    model_name = os.path.basename(pipeline.model_id)
    # get cached model path
    model_path = os.path.join("/root/.cache/gooey-gpu/checkpoints", model_name)
    # if not cached, download again
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(requests.get(pipeline.model_id).content)
    # save audio to tmp file
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(inputs.audio)[-1]) as f:
        f.write(requests.get(inputs.audio).content)
        # run ASR
        return {"text": run_nemo_asr(model_path, f.name)}


@gooey_gpu.gpu_task
def run_nemo_asr(model_path: str, audio_file: str) -> str:
    asr_model = load_model(model_path)
    with gooey_gpu.use_models(asr_model):
        return asr_model.transcribe(paths2audio_files=[audio_file])[0]


@lru_cache
def load_model(model_path):
    return nemo_asr.models.ASRModel.restore_from(model_path)
