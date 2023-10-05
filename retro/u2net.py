import os
import shutil
import sys
import typing
from functools import lru_cache

import PIL.Image
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms  # , utils

import gooey_gpu
from api import PipelineInfo
from celeryconfig import app, setup_queues

sys.path.append(os.path.join(os.path.dirname(__file__), "U-2-Net"))

from data_loader import RescaleT
from data_loader import SalObjDataset
from data_loader import ToTensorLab
from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB
from u2net_test import normPRED, save_output


@app.task(name="u2net")
@gooey_gpu.endpoint
def u2net(pipeline: PipelineInfo, inputs: typing.List[str]):
    net = load_model(pipeline.model_id)
    prediction_dir = "outputs/"

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(
        img_name_list=inputs,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]),
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("inferencing:", inputs[i_test].split(os.sep)[-1])

        inputs_test = data_test["image"]
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        shutil.rmtree(prediction_dir, ignore_errors=True)
        os.makedirs(prediction_dir, exist_ok=True)
        save_output(inputs[i_test], pred, prediction_dir)

        # upload image
        out_path = prediction_dir + os.listdir(prediction_dir)[0]
        gooey_gpu.upload_image(PIL.Image.open(out_path), pipeline.upload_urls[i_test])

        del d1, d2, d3, d4, d5, d6, d7


@lru_cache
def load_model(model_id):
    model_dir = os.path.join(
        gooey_gpu.CHECKPOINTS_DIR, "saved_models", model_id, f"{model_id}.pth"
    )

    if model_id == "u2net":
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_id == "u2netp":
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location="cpu"))
    net.eval()

    return net


setup_queues(
    model_ids=os.environ["U2NET_MODEL_IDS"].split(","),
    load_fn=load_model,
)
