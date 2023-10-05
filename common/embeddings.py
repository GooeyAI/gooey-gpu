import os
from functools import lru_cache

import torch
from pydantic import BaseModel
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import gooey_gpu
from api import PipelineInfo
from celeryconfig import app, setup_queues


class EmbeddingsInputs(BaseModel):
    texts: list[str]


@app.task(name="text_embeddings")
@gooey_gpu.endpoint
def text_embeddings(pipeline: PipelineInfo, inputs: EmbeddingsInputs):
    tokenizer, model = load_pipe(pipeline.model_id)
    with torch.inference_mode():
        # Tokenize the input texts
        batch_dict = tokenizer(
            inputs.texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(gooey_gpu.DEVICE_ID)

        outputs = model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        return embeddings.tolist()


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@lru_cache
def load_pipe(model_id: str):
    print(f"Loading embedding model {model_id!r}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(gooey_gpu.DEVICE_ID)
    return tokenizer, model


setup_queues(
    model_ids=os.environ["EMBEDDING_MODEL_IDS"].split(),
    load_fn=load_pipe,
)
