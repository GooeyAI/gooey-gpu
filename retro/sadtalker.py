import json
import mimetypes
import os
import subprocess
import sys
import typing
from functools import lru_cache
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from pydantic import BaseModel, HttpUrl
from skimage import img_as_ubyte
from tqdm import tqdm

import gooey_gpu
from celeryconfig import app, setup_queues

sadtalker_lib_path = os.path.join(os.path.dirname(__file__), "SadTalker")
sys.path.append(sadtalker_lib_path)

from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.test_audio2coeff import Audio2Coeff
from src.utils.init_path import init_path
from src.utils.preprocess import CropAndExtract
from src.facerender.modules.make_animation import keypoint_transformation


class SadtalkerPipeline(BaseModel):
    upload_urls: typing.List[HttpUrl]  # upload url for the output video
    model_id: str
    size: int = 512  # image size of the facerender
    preprocess: str = (
        "crop"  # how to preprocess the images, choices=['crop', 'extcrop', 'resize', 'full', 'extfull']
    )


class SadtalkerInput(BaseModel):
    source_image: HttpUrl  # source image url
    driven_audio: HttpUrl  # driven audio url
    pose_style: int = 0  # input pose style from [0, 46)
    ref_eyeblink: typing.Optional[HttpUrl] = (
        None  # reference video providing eye blinking
    )
    ref_pose: typing.Optional[HttpUrl] = None  # reference video providing pose
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


checkpoint_dir = os.path.join(gooey_gpu.CHECKPOINTS_DIR, "sadtalker")
try:
    os.symlink(
        os.path.join(checkpoint_dir, "gfpgan"), "gfpgan", target_is_directory=True
    )
except FileExistsError:
    pass


@lru_cache
def load_model(
    checkpoint: str, preprocess: str
) -> typing.Tuple[CropAndExtract, Audio2Coeff, AnimateFromCoeff]:
    print(f"loading {checkpoint} {preprocess}...")
    sadtalker_paths = init_path(
        checkpoint_dir=checkpoint_dir,
        config_dir=os.path.join(sadtalker_lib_path, "src/config"),
        preprocess=preprocess,
    )
    sadtalker_paths["checkpoint"] = os.path.join(checkpoint_dir, checkpoint)
    return (
        CropAndExtract(sadtalker_paths, gooey_gpu.DEVICE_ID),
        Audio2Coeff(sadtalker_paths, gooey_gpu.DEVICE_ID),
        AnimateFromCoeff(sadtalker_paths, gooey_gpu.DEVICE_ID),
    )


def load_model_all(model_id: str):
    load_model(model_id, "crop")
    load_model(model_id, "full")


setup_queues(
    model_ids=os.environ["SADTALKER_MODEL_IDS"].split(),
    load_fn=load_model_all,
)


@app.task(name="lipsync.sadtalker")
@gooey_gpu.endpoint
def sadtalker(pipeline: SadtalkerPipeline, inputs: SadtalkerInput) -> None:
    assert len(pipeline.upload_urls) == 1, "Expected exactly 1 upload url"

    face_mime_type = mimetypes.guess_type(inputs.source_image)[0] or ""
    if not ("video/" in face_mime_type or "image/" in face_mime_type):
        raise ValueError(f"Unsupported face format {face_mime_type!r}")

    audio_mime_type = mimetypes.guess_type(inputs.driven_audio)[0] or ""
    if not ("audio/" in audio_mime_type or "video/" in audio_mime_type):
        raise ValueError(f"Unsupported audio format {audio_mime_type!r}")

    with TemporaryDirectory() as save_dir:
        input_path, _ = urlretrieve(
            inputs.source_image,
            os.path.join(save_dir, "face" + os.path.splitext(inputs.source_image)[1]),
        )
        audio_path, _ = urlretrieve(
            inputs.driven_audio,
            os.path.join(save_dir, "audio" + os.path.splitext(inputs.driven_audio)[1]),
        )
        if audio_mime_type != "audio/wav":
            wav_audio_path = audio_path + ".wav"
            args = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                wav_audio_path,
            ]  # fmt:skip
            print("\t$ " + " ".join(args))
            print(subprocess.check_output(args, encoding="utf-8"))
            audio_path = wav_audio_path

        preprocess_model, audio_to_coeff, animate_from_coeff = load_model(
            pipeline.model_id, "full" if "full" in pipeline.preprocess else "crop"
        )

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            input_path,
            first_frame_dir,
            pipeline.preprocess,
            source_image_flag=True,
            pic_size=pipeline.size,
        )
        if first_coeff_path is None:
            raise ValueError("Can't get the coeffs of the input")

        if inputs.ref_eyeblink:
            ref_eyeblink, _ = urlretrieve(
                inputs.ref_eyeblink, os.path.join(save_dir, "ref_eyeblink.mp4")
            )
            ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
                ref_eyeblink,
                save_dir,
                pipeline.preprocess,
            )
        else:
            ref_eyeblink = None
            ref_eyeblink_coeff_path = None

        if inputs.ref_pose:
            ref_pose, _ = urlretrieve(
                inputs.ref_pose, os.path.join(save_dir, "ref_pose.mp4")
            )
            if ref_pose == ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_coeff_path, _, _ = preprocess_model.generate(
                    ref_pose,
                    save_dir,
                    pipeline.preprocess,
                )
        else:
            ref_pose_coeff_path = None

        # audio2ceoff
        batch = get_data(
            first_coeff_path,
            audio_path,
            gooey_gpu.DEVICE_ID,
            ref_eyeblink_coeff_path,
            still=inputs.still,
        )
        coeff_path = audio_to_coeff.generate(
            batch, save_dir, inputs.pose_style, ref_pose_coeff_path
        )

        # 3dface render
        if inputs.face3dvis:
            from src.face3d.visualize import gen_composed_video

            gen_composed_video(
                inputs,
                gooey_gpu.DEVICE_ID,
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
            1,  ## with batch size > 1, the frames are out of sync
            inputs.input_yaw,
            inputs.input_pitch,
            inputs.input_roll,
            expression_scale=inputs.expression_scale,
            still_mode=inputs.still,
            preprocess=pipeline.preprocess,
            size=pipeline.size,
        )

        result_path = animate_from_coeff_generate(
            animate_from_coeff,
            x=data,
            video_save_dir=save_dir,
            input_path=input_path,
            crop_info=crop_info,
            enhancer=inputs.enhancer,
            background_enhancer=inputs.background_enhancer,
            preprocess=pipeline.preprocess,
            img_size=pipeline.size,
        )

        with open(result_path, "rb") as f:
            gooey_gpu.upload_video_from_bytes(f.read(), pipeline.upload_urls[0])


def animate_from_coeff_generate(
    self,
    x,
    video_save_dir,
    input_path,
    crop_info,
    enhancer,
    background_enhancer,
    preprocess,
    img_size,
):

    source_image = x["source_image"].type(torch.FloatTensor)
    source_semantics = x["source_semantics"].type(torch.FloatTensor)
    target_semantics = x["target_semantics_list"].type(torch.FloatTensor)
    source_image = source_image.to(self.device)
    source_semantics = source_semantics.to(self.device)
    target_semantics = target_semantics.to(self.device)
    if "yaw_c_seq" in x:
        yaw_c_seq = x["yaw_c_seq"].to(self.device)
    else:
        yaw_c_seq = None
    if "pitch_c_seq" in x:
        pitch_c_seq = x["pitch_c_seq"].to(self.device)
    else:
        pitch_c_seq = None
    if "roll_c_seq" in x:
        roll_c_seq = x["roll_c_seq"].to(self.device)
    else:
        roll_c_seq = None

    video_name = x["video_name"] + ".mp4"
    return_path = os.path.join(video_save_dir, video_name)

    img_size = int(img_size) // 2 * 2
    original_size = crop_info[0]
    if original_size:
        frame_w, frame_h = (
            img_size,
            int(img_size * original_size[1] / original_size[0]) // 2 * 2,
        )
    else:
        frame_w, frame_h = (img_size, img_size)

    if "full" in preprocess.lower():
        input_frames, out_w, out_h = read_video_frames(input_path)

        if len(crop_info) != 3:
            raise ValueError("you didn't crop the image")
        else:
            clx, cly, crx, cry = crop_info[1]
            lx, ly, rx, ry = crop_info[2]
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            if "ext" in preprocess.lower():
                oy1, oy2, ox1, ox2 = cly, cry, clx, crx
            else:
                oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx
    else:
        out_w, out_h = frame_w, frame_h

    cmd_args = [
        "ffmpeg",
        # "-thread_queue_size", "128",
        "-pixel_format", "rgb24",
        "-f", "rawvideo",
        # "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-r", "25",
        "-i", "pipe:0",  # stdin
        "-i", x['audio_path'],
        # "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",  # because iphone, see https://trac.ffmpeg.org/wiki/Encode/H.264#Encodingfordumbplayers
        # "-preset", "ultrafast",
        return_path,
    ]  # fmt:skip
    print("\t$ " + " ".join(cmd_args))
    ffproc = subprocess.Popen(cmd_args, stdin=subprocess.PIPE)

    i = 0
    frame_num = x["frame_num"]
    for batch in make_animation(
        source_image,
        source_semantics,
        target_semantics,
        self.generator,
        self.kp_extractor,
        self.he_estimator,
        self.mapping,
        yaw_c_seq,
        pitch_c_seq,
        roll_c_seq,
    ):
        for out_image in batch:
            i += 1
            if i > frame_num:
                break
            out_image = img_as_ubyte(
                out_image.data.cpu().numpy().transpose([1, 2, 0]).astype(np.float32)
            )
            out_image = cv2.resize(out_image, (frame_w, frame_h))

            if "full" in preprocess.lower():
                input_image = input_frames[i % len(input_frames)]
                p = cv2.resize(out_image, (ox2 - ox1, oy2 - oy1))
                mask = 255 * np.ones(p.shape, p.dtype)
                location = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
                try:
                    out_image = cv2.seamlessClone(
                        p, input_image, mask, location, cv2.NORMAL_CLONE
                    )
                except cv2.error:
                    raise ValueError(
                        "Failed to perform full preprocess. Please use the crop mode or try a different aspect ratio."
                    )

            ffproc.stdin.write(out_image.tostring())

    ffproc.stdin.close()
    ffproc.wait()

    return return_path


def read_video_frames(
    input_path: str,
) -> typing.Tuple[typing.List[np.ndarray], int, int]:
    cmd_args = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams", input_path,
        "-select_streams", "v:0",
    ]  # fmt:skip
    print("\t$ " + " ".join(cmd_args))
    data = json.loads(subprocess.check_output(cmd_args, text=True))
    if not data["streams"]:
        raise ValueError("input video has no streams")
    out_w, out_h = (
        int(data["streams"][0]["width"]) // 2 * 2,
        int(data["streams"][0]["height"]) // 2 * 2,
    )

    cmd_args = [
        "ffmpeg",
        "-i", input_path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", "25",
        "pipe:1",
    ]  # fmt:skip
    print("\t$ " + " ".join(cmd_args))
    ffproc = subprocess.Popen(cmd_args, stdout=subprocess.PIPE)

    input_frames = []
    while True:
        raw_image = ffproc.stdout.read(out_h * out_w * 3)
        if not raw_image:
            break
        input_image = np.frombuffer(raw_image, dtype=np.uint8).reshape(
            (out_h, out_w, 3)
        )
        input_frames.append(input_image)

    return input_frames, out_w, out_h


def make_animation(
    source_image,
    source_semantics,
    target_semantics,
    generator,
    kp_detector,
    he_estimator,
    mapping,
    yaw_c_seq=None,
    pitch_c_seq=None,
    roll_c_seq=None,
    use_exp=True,
    use_half=False,
):
    with torch.no_grad():
        kp_canonical = kp_detector(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)

        for frame_idx in tqdm(range(target_semantics.shape[1]), "Face Renderer"):
            # still check the dimension
            # print(target_semantics.shape, source_semantics.shape)
            target_semantics_frame = target_semantics[:, frame_idx]
            he_driving = mapping(target_semantics_frame)
            if yaw_c_seq is not None:
                he_driving["yaw_in"] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving["pitch_in"] = pitch_c_seq[:, frame_idx]
            if roll_c_seq is not None:
                he_driving["roll_in"] = roll_c_seq[:, frame_idx]

            kp_driving = keypoint_transformation(kp_canonical, he_driving)

            kp_norm = kp_driving
            out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
            yield out["prediction"]
