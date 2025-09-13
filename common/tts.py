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

    # Build a language-specific pipeline from the downloaded repository
    language = inputs.language or "eng"
    pipe = load_pipe(pipeline.model_id, language)

    with torch.no_grad():
        result = pipe(inputs.text)

    # Normalize output format
    if isinstance(result, dict):
        audio_obj = result.get("audio", result)
        if isinstance(audio_obj, dict) and "array" in audio_obj:
            waveform = audio_obj["array"]
            sampling_rate = audio_obj.get("sampling_rate")
        else:
            waveform = np.asarray(audio_obj)
            sampling_rate = result.get("sampling_rate")
    else:
        waveform = np.asarray(result)
        sampling_rate = None

    if sampling_rate is None:
        try:
            sampling_rate = pipe.model.config.sampling_rate
        except Exception:
            sampling_rate = 22050

    for url in pipeline.upload_urls:
        gooey_gpu.upload_audio(waveform, url, rate=sampling_rate)


@lru_cache
def load_repo(model_id: str) -> str:
    print(f"Downloading repository {model_id!r} if not cached...")
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        from transformers.utils.hub import snapshot_download  # type: ignore
    repo_dir = snapshot_download(repo_id=model_id, allow_patterns=["models/*"])
    return repo_dir


@lru_cache
def load_pipe(model_id: str, language: str):
    repo_dir = load_repo(model_id)
    lang_dir = os.path.join(repo_dir, "models", language)
    print(f"Loading TTS pipeline for language {language!r} from {lang_dir!r}...")
    pipe = transformers.pipeline(
        "text-to-speech",
        model=lang_dir,
        device=gooey_gpu.DEVICE_ID,
        torch_dtype=torch.float16,
    )
    return pipe


setup_queues(
    model_ids=os.environ.get("TTS_MODEL_IDS", "facebook/mms-tts").split(),
    load_fn=load_repo,
)
