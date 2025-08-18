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
def load_model(model_id_or_url: str):
    print(f"Loading nemo asr model {model_id_or_url!r}...")
    if is_url(model_id_or_url):
        return load_model_from_url(model_id_or_url)
    else:
        return nemo_asr.models.ASRModel.from_pretrained(model_id_or_url)


def load_model_from_url(model_url: str):
    # get cached model path
    model_path = os.path.join(gooey_gpu.CHECKPOINTS_DIR, os.path.basename(model_url))
    # if not cached, download again
    gooey_gpu.download_file_to_path(url=model_url, path=model_path, cached=True)
    # load model
    return nemo_asr.models.ASRModel.restore_from(model_path)


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


setup_queues(
    model_ids=os.environ["NEMO_ASR_MODEL_IDS"].split(),
    load_fn=load_model,
)
