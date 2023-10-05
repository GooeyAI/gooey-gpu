import os.path
import tempfile
from functools import lru_cache

import nemo.collections.asr as nemo_asr
import requests

import gooey_gpu
from api import NemoASRInputs
from api import PipelineInfo, AsrOutput
from celeryconfig import app, setup_queues


@app.task(name="nemo_asr")
@gooey_gpu.endpoint
def nemo_asr_api(pipeline: PipelineInfo, inputs: NemoASRInputs) -> AsrOutput:
    # load model
    asr_model = load_model(pipeline.model_id)
    # save audio to tmp file
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(inputs.audio)[-1]) as f:
        f.write(requests.get(inputs.audio).content)
        # run ASR
        text = asr_model.transcribe(paths2audio_files=[f.name])[0]
    return AsrOutput(text=text)


@lru_cache
def load_model(model_id: str):
    print(f"Loading nemo asr model {model_id!r}...")
    model_name = os.path.basename(model_id)
    # get cached model path
    model_path = os.path.join(gooey_gpu.CHECKPOINTS_DIR, model_name)
    # if not cached, download again
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(requests.get(model_id).content)
    # load model
    return nemo_asr.models.ASRModel.restore_from(model_path)


setup_queues(
    model_ids=os.environ["NEMO_ASR_MODEL_IDS"].split(),
    load_fn=load_model,
)
