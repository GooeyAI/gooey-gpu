import os
import typing
from functools import lru_cache

import requests
import torch
import transformers
from pydantic import BaseModel

import io
from scipy.io.wavfile import write

import gooey_gpu
from api import AsrOutput
from celeryconfig import app, setup_queues


class SeamlessM4TPipeline(BaseModel):
    upload_urls: typing.List[str] = []
    model_id: typing.Literal["facebook/seamless-m4t-v2-large"] = (
        "facebook/seamless-m4t-v2-large"
    )


class SeamlessM4TInputs(BaseModel):
    audio: str | None = None  # required for ASR, S2ST, and S2TT
    text: str | None = None  # required for T2ST and T2TT
    src_lang: str | None = None  # required for T2ST and T2TT
    tgt_lang: str | None = None  # ignored for ASR (only src_lang is used)
    # seamless uses ISO 639-3 codes for languages

    chunk_length_s: float = 30
    stride_length_s: typing.Tuple[float, float] = (6, 0)
    batch_size: int = 16

    speaker_id: int = 0  # only used for T2ST, value in [0, 200)


@app.task(name="seamless.t2st")
@gooey_gpu.endpoint
def seamless_text2speech_translation(
    pipeline: SeamlessM4TPipeline,
    inputs: SeamlessM4TInputs,
) -> None:
    _, processor, model = load_pipe(pipeline.model_id)
    tgt_lang = inputs.tgt_lang or inputs.src_lang or "eng"

    assert inputs.text is not None
    assert inputs.src_lang is not None
    text_inputs = processor(
        text=inputs.text, src_lang=inputs.src_lang, return_tensors="pt"
    )

    audio_array_from_text = (
        model.generate(**text_inputs, tgt_lang=tgt_lang, speaker_id=inputs.speaker_id)[
            0
        ]
        .cpu()
        .numpy()
        .squeeze()
    )

    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, 16000, audio_array_from_text)
    audio_bytes = byte_io.read()
    gooey_gpu.upload_audio_from_bytes(audio_bytes, pipeline.upload_urls[0])
    return


@app.task(name="seamless.t2tt")
@gooey_gpu.endpoint
def seamless_text2text_translation(
    pipeline: SeamlessM4TPipeline,
    inputs: SeamlessM4TInputs,
) -> AsrOutput | None:
    _, processor, model = load_pipe(pipeline.model_id)
    tgt_lang = inputs.tgt_lang or inputs.src_lang or "eng"

    assert inputs.text is not None
    assert inputs.src_lang is not None
    text_inputs = processor(
        text=inputs.text, src_lang=inputs.src_lang, return_tensors="pt"
    )

    output_tokens = model.generate(
        **text_inputs, tgt_lang=tgt_lang, generate_speech=False
    )
    translated_text_from_text = processor.decode(
        output_tokens[0].tolist()[0], skip_special_tokens=True
    )

    return AsrOutput(text=translated_text_from_text)


@app.task(name="seamless")
@gooey_gpu.endpoint
def seamless_asr(
    pipeline: SeamlessM4TPipeline,
    inputs: SeamlessM4TInputs,
) -> AsrOutput | None:
    pipe, _, _ = load_pipe(pipeline.model_id)
    tgt_lang = inputs.tgt_lang or inputs.src_lang or "eng"

    assert inputs.audio is not None

    audio = requests.get(inputs.audio).content

    previous_src_lang = pipe.tokenizer.src_lang
    if inputs.src_lang:
        pipe.tokenizer.src_lang = inputs.src_lang

    prediction = pipe(
        audio,
        # see https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor#scrollTo=Ca4YYdtATxzo&line=5&uniqifier=1
        chunk_length_s=inputs.chunk_length_s,
        stride_length_s=inputs.stride_length_s,
        batch_size=inputs.batch_size,
        generate_kwargs=dict(tgt_lang=tgt_lang),
    )

    pipe.tokenizer.src_lang = previous_src_lang

    return prediction


@lru_cache
def load_pipe(
    model_id: str,
) -> typing.Tuple[
    transformers.AutomaticSpeechRecognitionPipeline,
    transformers.SeamlessM4TProcessor,
    transformers.SeamlessM4Tv2Model,
]:
    print(f"Loading asr model {model_id!r}...")
    pipe = transformers.pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=gooey_gpu.DEVICE_ID,
        torch_dtype=torch.float16,
    )
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    model = transformers.SeamlessM4Tv2Model.from_pretrained(model_id)
    return pipe, processor, model


setup_queues(
    model_ids=os.environ["SEAMLESS_MODEL_IDS"].split(),
    load_fn=load_pipe,
)
