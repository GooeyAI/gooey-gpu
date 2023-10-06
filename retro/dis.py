import os
import sys
import typing
from functools import lru_cache

import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from torchvision.transforms.functional import normalize
from tqdm import tqdm

import gooey_gpu
from api import PipelineInfo
from celeryconfig import app, setup_queues

sys.path.append(os.path.join(os.path.dirname(__file__), "DIS", "IS-Net"))

from models import ISNetDIS


@app.task(name="dis")
@gooey_gpu.endpoint
def dis(pipeline: PipelineInfo, inputs: typing.List[str]):
    """Run a single prediction on the model"""
    net = setup(pipeline.model_id)
    try:
        os.remove("out.png")
    except FileNotFoundError:
        pass
    input_size = [1024, 1024]
    for i, im_path in tqdm(enumerate(inputs), total=len(inputs)):
        print("im_path: ", im_path)
        im = io.imread(im_path)
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_shp = im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.upsample(
            torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear"
        ).type(torch.uint8)
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

        if torch.cuda.is_available():
            image = image.cuda()
        result = net(image)
        result = torch.squeeze(F.upsample(result[0][0], im_shp, mode="bilinear"), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        im_pil = PIL.Image.fromarray(im)
        gooey_gpu.upload_image(im_pil, pipeline.upload_urls[i])


@lru_cache
def setup(model_id):
    """Load the model into memory to make running multiple predictions efficient"""
    model_path = os.path.join(gooey_gpu.CHECKPOINTS_DIR, model_id)
    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    return net


setup_queues(
    model_ids=os.environ["DIS_MODEL_IDS"].split(","),
    load_fn=setup,
)
