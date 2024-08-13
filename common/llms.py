import os
import typing
from functools import lru_cache

import torch
import transformers
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers.models.auto.tokenization_auto import get_tokenizer_config

import gooey_gpu
from celeryconfig import app, setup_queues


class PipelineInfo(BaseModel):
    model_id: str
    seed: int = None
    fallback_chat_template_from: str | None


class LLMChatInputs(BaseModel):
    text_inputs: typing.List[dict] | str
    max_new_tokens: int
    stop_strings: typing.Optional[typing.List[str]]
    temperature: float = 1
    repetition_penalty: float = 1


class LLMChatOutput(BaseModel):
    generated_text: str
    usage: dict


@app.task(name="llm.chat")
@gooey_gpu.endpoint
def llm_chat(pipeline: PipelineInfo, inputs: LLMChatInputs) -> LLMChatOutput:
    pipe = load_pipe(pipeline.model_id)

    if pipeline.fallback_chat_template_from and not pipe.tokenizer.chat_template:
        # if the tokenizer does not have a chat template, use the provided fallback
        config = get_tokenizer_config(pipeline.fallback_chat_template_from)
        pipe.tokenizer.chat_template = config.get("chat_template")

    # for a list of parameters, see https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
    ret = pipe(
        inputs.text_inputs,
        max_new_tokens=inputs.max_new_tokens,
        stop_strings=inputs.stop_strings,
        temperature=inputs.temperature,
        repetition_penalty=inputs.repetition_penalty,
        tokenizer=pipe.tokenizer,
        do_sample=True,
        eos_token_id=pipe.tokenizer.eos_token_id,
    )[0]

    # strip stop strings & eos token from final output
    for s in (inputs.stop_strings or []) + [pipe.tokenizer.eos_token]:
        ret["generated_text"] = ret["generated_text"].split(s, 1)[0]

    return ret


@lru_cache
def load_pipe(model_id: str) -> transformers.TextGenerationPipeline:
    print(f"Loading llm model {model_id!r}...")
    # this should return a TextGenerationPipeline
    pipe = typing.cast(
        transformers.TextGenerationPipeline,
        transformers.pipeline(
            "text-generation",
            model=model_id,
            device=gooey_gpu.DEVICE_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ),
    )
    if not pipe.tokenizer:
        pipe.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def postprocess_new(
        model_outputs, return_type=None, clean_up_tokenization_spaces=True
    ):
        generated_sequence = model_outputs["generated_sequence"][0]
        input_ids = model_outputs["input_ids"]
        # prompt_text = model_outputs["prompt_text"]
        generated_sequence = generated_sequence.numpy().tolist()
        records = []
        for sequence in generated_sequence:
            # Decode text
            text = pipe.tokenizer.decode(
                sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

            # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
            if input_ids is None:
                prompt_length = prompt_tokens = 0
            else:
                prompt_tokens = len(input_ids[0])
                prompt = pipe.tokenizer.decode(
                    input_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
                prompt_length = len(prompt)

            records.append(
                {
                    "usage": {
                        "completion_tokens": len(sequence) - prompt_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": len(sequence),
                    },
                    "generated_text": text[prompt_length:],
                }
            )

        return records

    _postprocess_old = pipe.postprocess
    pipe.postprocess = postprocess_new

    return pipe


setup_queues(
    model_ids=os.environ["LLM_MODEL_IDS"].split(),
    load_fn=load_pipe,
)
