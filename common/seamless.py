import os
from functools import lru_cache
import tempfile
from pydub import AudioSegment
import math
import textwrap
import torchaudio

import requests
import torch
from celery.signals import worker_init
from kombu import Queue
from seamless_communication.models.inference import Translator

import gooey_gpu
from api import (
    SeamlessM4TPipeline,
    SeamlessM4TInputs,
    SeamlessM4TOutput,
)
from celeryconfig import app, setup_queues

VOCODER_ID = "vocoder_36langs"  # only supported option rn


@app.task(name="seamless")
@gooey_gpu.endpoint
def seamless(
    pipeline: SeamlessM4TPipeline,
    inputs: SeamlessM4TInputs,
) -> SeamlessM4TOutput:
    with tempfile.NamedTemporaryFile("br+") as temp_file:
        audio_path = input_text = translator = None
        match inputs.task:
            case "ASR" | "S2ST" | "S2TT":
                audio = requests.get(inputs.audio).content
                temp_file.write(audio)
                audio_path = temp_file.name
                translator = load_translator(pipeline.model_id)
            case "T2ST" | "T2TT":
                input_text = inputs.text
                translator = load_translator(pipeline.model_id)
            case _:
                raise Exception("Unsupported task for SeamlessM4T")

        # we need chunking because seamless has a max sequence length of 4096
        texts = []
        wavs = []
        chunks = (
            _chunked_text(input_text, 4096)
            if input_text
            else _chunked_audio_file(audio_path, 1)
        )
        for chunk in chunks:
            text, wav, sr = _seamless_one_chunk(
                inputs.task, chunk, translator, inputs.tgt_lang, inputs.src_lang
            )
            if text:
                texts.append(text)
            if wav is not None:
                wavs.append((wav, sr))

        with tempfile.NamedTemporaryFile("br+") as outfile:
            audio_url = None
            if wavs:
                _combine_wavs(wavs, outfile)
                audio_bytes = outfile.read()
                audio_url = pipeline.upload_urls[0]
                gooey_gpu.upload_audio_from_bytes(audio_bytes, audio_url)
            return SeamlessM4TOutput(text="\n\n".join(map(str, texts)), audio=audio_url)


def _seamless_one_chunk(task, path_or_text, translator, tgt_lang, src_lang):
    # follows these instructions: https://huggingface.co/facebook/seamless-m4t-large
    text = wav = sr = None
    match task:
        case "S2ST":
            text, wav, sr = translator.predict(
                path_or_text,
                task.lower(),
                tgt_lang,
            )
        case "T2ST":
            text, wav, sr = translator.predict(
                path_or_text,
                task.lower(),
                tgt_lang,
                src_lang=src_lang,
            )
        case "S2TT" | "ASR":
            text, _, _ = translator.predict(
                path_or_text,
                task.lower(),
                (src_lang if task == "ASR" else tgt_lang),
            )
        case "T2TT":
            text, _, _ = translator.predict(
                path_or_text,
                task.lower(),
                tgt_lang,
                src_lang=src_lang,
            )
    return text, wav, sr


@lru_cache
def load_translator(model_id: str):
    print(f"Loading seamless model {model_id!r}...")
    # Initialize a Translator object with a multitask model, vocoder
    return Translator(
        model_id,
        vocoder_name_or_card=VOCODER_ID,
        device=torch.device(gooey_gpu.DEVICE_ID),
    )


setup_queues(
    model_ids=os.environ["SEAMLESS_MODEL_IDS"].split(),
    load_fn=load_translator,
)


def _chunked_audio_file(file, min_per_split: int):
    audio: AudioSegment = AudioSegment.from_wav(file)
    total_mins = math.ceil(audio.duration_seconds / 60)
    for i in range(0, total_mins, min_per_split):
        with tempfile.NamedTemporaryFile("br+") as temp_file:
            t1 = i * 60 * 1000
            t2 = (i + min_per_split) * 60 * 1000
            audio[t1:t2].export(temp_file.name, format="wav")
            yield temp_file.name


def _chunked_text(text: str, tokens_per_split):
    return textwrap.wrap(text, 3 * tokens_per_split)


def _combine_wavs(wavs, outfile):
    combined_sound = AudioSegment.empty()
    for tensor, sr in wavs:
        with tempfile.NamedTemporaryFile("wb", suffix=".wav") as wav:
            torchaudio.save(
                wav.name,
                tensor[0].cpu(),
                sample_rate=sr,
                encoding="PCM_S",
                bits_per_sample=16,
            )
            combined_sound += AudioSegment.from_wav(wav.name)
    combined_sound.export(outfile.name, format="wav")
