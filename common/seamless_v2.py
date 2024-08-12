import typing
from functools import lru_cache

import transformers
from pydantic import BaseModel

import gooey_gpu
from celeryconfig import app


class SeamlessPipeline(BaseModel):
    upload_urls: typing.List[str] = []
    model_id: str


class SeamlessT2STInputs(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    speaker_id: int = 0  # [0, 200)


@app.task(name="seamless.t2st")
@gooey_gpu.endpoint
def seamless_text_to_speech_translation(
    pipeline: SeamlessPipeline,
    inputs: SeamlessT2STInputs,
) -> None:
    model, processor = load_model(pipeline.model_id)
    text_inputs = processor(
        text=inputs.text, src_lang=inputs.src_lang, return_tensors="pt"
    ).to(gooey_gpu.DEVICE_ID)
    audio_array_from_text = (
        model.generate(
            **text_inputs, tgt_lang=inputs.tgt_lang, speaker_id=inputs.speaker_id
        )[0]
        .cpu()
        .numpy()
        .squeeze()
    )
    gooey_gpu.upload_audio(audio_array_from_text, pipeline.upload_urls[0])


class SeamlessT2TTInputs(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str


@app.task(name="seamless.t2tt")
@gooey_gpu.endpoint
def seamless_text2text_translation(
    pipeline: SeamlessPipeline,
    inputs: SeamlessT2TTInputs,
) -> str:
    model, processor = load_model(pipeline.model_id)
    text_inputs = processor(
        text=inputs.text, src_lang=inputs.src_lang, return_tensors="pt"
    ).to(gooey_gpu.DEVICE_ID)
    output_tokens = model.generate(
        **text_inputs, tgt_lang=inputs.tgt_lang, generate_speech=False
    )
    translated_text_from_text = processor.decode(
        output_tokens[0].tolist()[0], skip_special_tokens=True
    )
    return translated_text_from_text


@lru_cache
def load_model(model_id: str) -> typing.Tuple[
    transformers.SeamlessM4Tv2Model,
    transformers.SeamlessM4TProcessor,
]:
    print(f"Loading seamless m4t model {model_id!r}...")
    model = typing.cast(
        transformers.SeamlessM4Tv2Model,
        transformers.AutoModel.from_pretrained(model_id).to(gooey_gpu.DEVICE_ID),
    )
    processor = typing.cast(
        transformers.SeamlessM4TProcessor,
        transformers.AutoProcessor.from_pretrained(model_id),
    )
    return model, processor
