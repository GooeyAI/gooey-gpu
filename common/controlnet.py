import PIL.Image
import controlnet_aux
import cv2
import numpy as np
import torch
import transformers
from diffusers import (
    StableDiffusionControlNetPipeline,
)
from fastapi import APIRouter
from pydantic.tools import lru_cache
from starlette.requests import Request

import gooey_gpu
from api import ControlNetPipelineInfo, ControlNetInputs, MAX_IMAGE_SIZE
from common.diffusion import predict_and_upload

app = APIRouter()


@app.post("/controlnet/")
@gooey_gpu.endpoint
def controlnet(
    request: Request, pipeline: ControlNetPipelineInfo, inputs: ControlNetInputs
):
    image = gooey_gpu.download_images(inputs.image, MAX_IMAGE_SIZE)
    if not pipeline.disable_preprocessing:
        for idx, (im, controlnet_model_id) in enumerate(
            zip(image, pipeline.controlnet_model_id)
        ):
            try:
                preprocessor = CONTROLNET_PREPROCESSORS[pipeline.controlnet_model_id]
            except KeyError:
                pass
            else:
                image[idx] = preprocessor(im)
    return predict_and_upload(
        request=request,
        pipe_cls=StableDiffusionControlNetPipeline,
        pipeline=pipeline,
        inputs=inputs,
        inputs_mod=dict(
            image=image,
        ),
    )


def canny(im_pil):
    image = np.array(im_pil)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return PIL.Image.fromarray(image)


def depth(im_pil):
    depth_estimator = load_depth_estimator()
    image = depth_estimator(im_pil)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return PIL.Image.fromarray(image)


@lru_cache
def load_depth_estimator():
    return transformers.pipeline("depth-estimation", device=gooey_gpu.DEVICE_ID)


def hed(im_pil):
    model = load_hed()
    return model(im_pil)


@lru_cache
def load_hed():
    return controlnet_aux.HEDdetector.from_pretrained("lllyasviel/ControlNet")


def scribble(im_pil):
    model = load_controlnet()
    return model(im_pil, scribble=True)


@lru_cache
def load_controlnet():
    return controlnet_aux.HEDdetector.from_pretrained("lllyasviel/ControlNet")


def mlsd(im_pil):
    model = load_mlsd()
    return model(im_pil)


@lru_cache
def load_mlsd():
    return controlnet_aux.MLSDdetector.from_pretrained("lllyasviel/ControlNet")


def openpose(im_pil):
    model = load_openpose()
    return model(im_pil)


@lru_cache
def load_openpose():
    return controlnet_aux.OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


def seg(im_pil):
    image_processor, image_segmentor = load_seg()

    pixel_values = image_processor(im_pil, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[im_pil.size[::-1]]
    )[0]

    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3

    palette = np.array(ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)

    return PIL.Image.fromarray(color_seg)


@lru_cache
def load_seg():
    image_processor = transformers.AutoImageProcessor.from_pretrained(
        "openmmlab/upernet-convnext-small"
    )
    image_segmentor = transformers.UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-small"
    )
    return image_processor, image_segmentor


def normal(im_pil):
    depth_estimator = load_normal()

    image = depth_estimator(im_pil)["predicted_depth"][0]

    image = image.numpy()

    image_depth = image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)

    bg_threhold = 0.4

    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0

    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0

    z = np.ones_like(x) * np.pi * 2.0

    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    return PIL.Image.fromarray(image)


@lru_cache
def load_normal():
    return transformers.pipeline(
        "depth-estimation", model="Intel/dpt-hybrid-midas", device=gooey_gpu.DEVICE_ID
    )


CONTROLNET_PREPROCESSORS = {
    "lllyasviel/sd-controlnet-canny": canny,
    "lllyasviel/sd-controlnet-depth": depth,
    "lllyasviel/sd-controlnet-hed": hed,
    "lllyasviel/sd-controlnet-mlsd": mlsd,
    "lllyasviel/sd-controlnet-normal": normal,
    "lllyasviel/sd-controlnet-openpose": openpose,
    "lllyasviel/sd-controlnet-scribble": scribble,
    "lllyasviel/sd-controlnet-seg": seg,
}


def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]
