import os
import typing
from functools import lru_cache

import torch
import transformers
from pydantic import BaseModel
from transformers import AutoTokenizer

import gooey_gpu
from celeryconfig import app, setup_queues


class PipelineInfo(BaseModel):
    model_id: str
    seed: int = None


class LLMChatInputs(BaseModel):
    messages: typing.List[dict]
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
    return pipe(
        inputs.messages,
        max_new_tokens=inputs.max_new_tokens,
        stop_strings=inputs.stop_strings,
        temperature=inputs.temperature,
        repetition_penalty=inputs.repetition_penalty,
        tokenizer=pipe.tokenizer,
        do_sample=True,
        eos_token_id=pipe.tokenizer.eos_token_id,
    )[0]


@lru_cache
def load_pipe(model_id: str):
    print(f"Loading llm model {model_id!r}...")
    # this should return a TextGenerationPipeline
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        device=gooey_gpu.DEVICE_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
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
                    "generated_text": text[prompt_length:],
                    "usage": {
                        "completion_tokens": len(sequence) - prompt_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": len(sequence),
                    },
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
