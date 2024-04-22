import gooey_gpu
from celeryconfig import app, setup_queues

import typing
from pydantic import BaseModel, HttpUrl

from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

# sadtalker
import torch
import os, sys

sys.path.insert(1, "./sadtalker/SadTalker")
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path


MODEL_ID = "sadtalker"
setup_queues(
    model_ids=[MODEL_ID],
    load_fn=lambda model_id: None,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_DIR = "./sadtalker/SadTalker/src/config"


class SadtalkerPipeline(BaseModel):
    model_id: typing.Literal[MODEL_ID] = MODEL_ID
    upload_urls: typing.List[HttpUrl]  # upload url for the output video


class SadtalkerInput(BaseModel):
    source_image: HttpUrl  # source image url
    driven_audio: HttpUrl  # driven audio url
    pose_style: int  # input pose style from [0, 46)
    ref_eyeblink: typing.Optional[HttpUrl] = (
        None  # reference video providing eye blinking
    )
    ref_pose: typing.Optional[HttpUrl] = None  # reference video providing pose
    batch_size: int = 2  # batch size of facerender
    size: int = 256  # image size of the facerender
    expression_scale: float = 1.0  # the batch size of facerender
    input_yaw: typing.Optional[typing.List[int]] = (
        None  # the input yaw degree of the user
    )
    input_pitch: typing.Optional[typing.List[int]] = (
        None  # the input pitch degree of the user
    )
    input_roll: typing.Optional[typing.List[int]] = (
        None  # the input roll degree of the user
    )
    enhancer: typing.Optional[str] = None  # Face enhancer, [gfpgan, RestoreFormer]
    background_enhancer: typing.Optional[str] = (
        None  # background enhancer, [realesrgan]
    )
    face3dvis: bool = False  # generate 3d face and 3d landmarks
    still: bool = (
        False  # can crop back to the original videos for the full body aniamtion
    )
    preprocess: str = (
        "crop"  # how to preprocess the images, choices=['crop', 'extcrop', 'resize', 'full', 'extfull']
    )


@app.task(name="lipsync.sadtalker")
@gooey_gpu.endpoint
def sadtalker(pipeline: SadtalkerPipeline, inputs: SadtalkerInput) -> None:
    assert len(pipeline.upload_urls) == 1, "Expected exactly 1 upload url"

    with TemporaryDirectory() as save_dir:
        pic_path, _ = urlretrieve(
            inputs.source_image, os.path.join(save_dir, "source_image.jpg")
        )
        audio_path, _ = urlretrieve(
            inputs.driven_audio, os.path.join(save_dir, "driven_audio.wav")
        )
        pose_style = inputs.pose_style
        batch_size = inputs.batch_size
        input_yaw_list = inputs.input_yaw
        input_pitch_list = inputs.input_pitch
        input_roll_list = inputs.input_roll
        ref_eyeblink = (
            urlretrieve(inputs.ref_eyeblink, os.path.join(save_dir, "ref_eyeblink.mp4"))
            if inputs.ref_eyeblink
            else None
        )
        ref_pose = (
            urlretrieve(inputs.ref_pose, os.path.join(save_dir, "ref_pose.mp4"))
            if inputs.ref_pose
            else None
        )

        # init model
        sadtalker_paths = init_path(
            "./checkpoints",
            CONFIG_DIR,
            inputs.size,
            False,
            inputs.preprocess,
        )
        preprocess_model = CropAndExtract(sadtalker_paths, DEVICE)
        audio_to_coeff = Audio2Coeff(sadtalker_paths, DEVICE)
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, DEVICE)

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            pic_path,
            first_frame_dir,
            inputs.preprocess,
            source_image_flag=True,
            pic_size=inputs.size,
        )
        if first_coeff_path is None:
            raise ValueError("Can't get the coeffs of the input")

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[
                0
            ]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
                ref_eyeblink,
                ref_eyeblink_frame_dir,
                inputs.preprocess,
                source_image_flag=False,
            )
        else:
            ref_eyeblink_coeff_path = None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                ref_pose_coeff_path, _, _ = preprocess_model.generate(
                    ref_pose,
                    ref_pose_frame_dir,
                    inputs.preprocess,
                    source_image_flag=False,
                )
        else:
            ref_pose_coeff_path = None

        # audio2ceoff
        batch = get_data(
            first_coeff_path,
            audio_path,
            DEVICE,
            ref_eyeblink_coeff_path,
            still=inputs.still,
        )
        coeff_path = audio_to_coeff.generate(
            batch, save_dir, pose_style, ref_pose_coeff_path
        )

        # 3dface render
        if inputs.face3dvis:
            from src.face3d.visualize import gen_composed_video

            gen_composed_video(
                inputs,
                DEVICE,
                first_coeff_path,
                coeff_path,
                audio_path,
                os.path.join(save_dir, "3dface.mp4"),
            )

        # coeff2video
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            input_yaw_list,
            input_pitch_list,
            input_roll_list,
            expression_scale=inputs.expression_scale,
            still_mode=inputs.still,
            preprocess=inputs.preprocess,
            size=inputs.size,
        )

        result_path = animate_from_coeff.generate(
            data,
            save_dir,
            pic_path,
            crop_info,
            enhancer=inputs.enhancer,
            background_enhancer=inputs.background_enhancer,
            preprocess=inputs.preprocess,
            img_size=inputs.size,
        )

        with open(result_path, "rb") as f:
            gooey_gpu.upload_video_from_bytes(f.read(), pipeline.upload_urls[0])
