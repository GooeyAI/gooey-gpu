import torch
from fastapi import APIRouter
from lavis.models import load_model_and_preprocess

import gooey_gpu
from api import PipelineInfo, VQAInput, MAX_IMAGE_SIZE, ImageCaptioningInput

app = APIRouter()


@app.post("/vqa/")
@gooey_gpu.endpoint
def vqa(pipeline: PipelineInfo, inputs: VQAInput):
    # load model
    model_id = pipeline.model_id.split("/")
    model, vis_processors, txt_processors = load_lavis_model(*model_id)
    # get inputs
    inputs_kwargs = inputs.dict()
    image = gooey_gpu.download_images(inputs_kwargs.pop("image"), MAX_IMAGE_SIZE)
    question = inputs_kwargs.pop("question")
    # do inference
    with gooey_gpu.use_models(model):
        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = torch.stack([vis_processors["eval"](im) for im in image]).to(
            gooey_gpu.DEVICE_ID
        )
        question = [txt_processors["eval"](q) for q in question]
        # generate answerss
        return model.predict_answers(
            samples={"image": image, "text_input": question}, **inputs_kwargs
        )
        # ['singapore']


@app.post("/image-captioning/")
@gooey_gpu.endpoint
def image_captioning(pipeline: PipelineInfo, inputs: ImageCaptioningInput):
    # load model
    model_id = pipeline.model_id.split("/")
    model, vis_processors, txt_processors = load_lavis_model(*model_id)
    # get inputs
    inputs_kwargs = inputs.dict()
    image = gooey_gpu.download_images(inputs_kwargs.pop("image"), MAX_IMAGE_SIZE)
    # do inference
    with gooey_gpu.use_models(model):
        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = torch.stack([vis_processors["eval"](im) for im in image]).to(
            gooey_gpu.DEVICE_ID
        )
        # generate caption
        return model.generate(samples={"image": image}, **inputs_kwargs)
        # ['a large fountain spewing water into the air']


_lavis_cache = {}


def load_lavis_model(name, model_type):
    try:
        ret = _lavis_cache[(name, model_type)]
    except KeyError:
        ret = load_model_and_preprocess(name, model_type, is_eval=True)
        _lavis_cache[(name, model_type)] = ret
    return ret
