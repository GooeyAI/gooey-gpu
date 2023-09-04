import argparse
import mimetypes
import os
import subprocess

import PIL.Image
import cv2
import numpy as np
import torch
from batch_face import RetinaFace
from tqdm import tqdm

import audio
import gooey_gpu
# from face_detect import face_rect
from models import Wav2Lip

parser = argparse.ArgumentParser(
    description="Inference code to lip-sync videos in the wild using Wav2Lip models"
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Name of saved checkpoint to load weights from",
    required=True,
)

parser.add_argument(
    "--face",
    type=str,
    help="Filepath of video/image that contains faces to use",
    required=True,
)
parser.add_argument(
    "--audio",
    type=str,
    help="Filepath of video/audio file to use as raw audio source",
    required=True,
)
parser.add_argument(
    "--outfile",
    type=str,
    help="Video path to save result. See default for an e.g.",
    default="results/result_voice.mp4",
)

parser.add_argument(
    "--static",
    type=bool,
    help="If True, then use only first video frame for inference",
    default=False,
)
parser.add_argument(
    "--fps",
    type=float,
    help="Can be specified only if input is a static image (default: 25)",
    default=25.0,
    required=False,
)

parser.add_argument(
    "--pads",
    nargs="+",
    type=int,
    default=[0, 10, 0, 0],
    help="Padding (top, bottom, left, right). Please adjust to include chin at least",
)

parser.add_argument(
    "--wav2lip_batch_size",
    type=int,
    help="Batch size for Wav2Lip model(s)",
    default=128,
)

# parser.add_argument('--resize_factor', default=1, type=int,
#             help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument(
    "--out_height",
    default=480,
    type=int,
    help="Output video height. Best results are obtained at 480 or 720",
)

parser.add_argument(
    "--crop",
    nargs="+",
    type=int,
    default=[0, -1, 0, -1],
    help="Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. "
    "Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width",
)

parser.add_argument(
    "--box",
    nargs="+",
    type=int,
    default=[-1, -1, -1, -1],
    help="Specify a constant bounding box for the face. Use only as a last resort if the face is not detected."
    "Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).",
)

parser.add_argument(
    "--rotate",
    default=False,
    action="store_true",
    help="Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg."
    "Use if you get a flipped result, despite feeding a normal looking video",
)

parser.add_argument(
    "--nosmooth",
    default=False,
    action="store_true",
    help="Prevent smoothing face detections over a short temporal window",
)


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T :]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    results = []
    pady1, pady2, padx1, padx2 = args.pads

    for image, faces in zip(images, detector(images)):
        if not faces:
            raise FaceNotFoundException(
                "Face not detected! Ensure the video contains a face in all the frames."
            )

        box, landmarks, score = faces[0]
        rect = tuple(map(int, box))

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    # if not args.nosmooth:
    #     boxes = get_smoothened_boxes(boxes, T=5)

    return boxes
    # results = [
    #     [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
    #     for image, (x1, y1, x2, y2) in zip(images, boxes)
    # ]
    #
    # return results


class FaceNotFoundException(ValueError):
    pass


mel_step_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} for inference.".format(device))


def _load(checkpoint_path):
    if device == "cuda":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def main():
    lip_size = 96

    face_mime_type = mimetypes.guess_type(args.face)[0] or ""
    args.static = "image/" in face_mime_type

    if not os.path.isfile(args.face):
        raise ValueError("--face argument must be a valid path to video/image file")

    if args.static:
        input_stream = None
        fps = args.fps
        frame = cv2.cvtColor(np.array(PIL.Image.open(args.face)), cv2.COLOR_RGB2BGR)
        frame = resize_frame(frame)
    else:
        input_stream = cv2.VideoCapture(args.face)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame = None

    ffproc = None

    mel_chunks = get_mel_chunks(fps)
    for idx in tqdm(range(0, len(mel_chunks), args.wav2lip_batch_size)):
        if args.static:
            frame_batch = [frame] * args.wav2lip_batch_size
        else:
            frame_batch = list(
                read_n_frames(idx, input_stream, args.wav2lip_batch_size)
            )

        if idx == 0:
            frame_h, frame_w = frame_batch[0].shape[:-1]
            cmd_args = [
                "ffmpeg",
                "-pixel_format", "bgr24",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{frame_w}x{frame_h}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-i", args.audio,
                "-vcodec", "libx264",
                "-preset", "ultrafast",
                args.outfile,
            ]  # fmt:skip
            print("\t$ " + " ".join(cmd_args))
            ffproc = subprocess.Popen(cmd_args, stdin=subprocess.PIPE)

        mel_batch = mel_chunks[idx : idx + args.wav2lip_batch_size]
        frame_batch = frame_batch[: len(mel_batch)]

        coords_batch = face_detect(frame_batch)
        img_batch = [
            cv2.resize(image[y1:y2, x1:x2], (lip_size, lip_size))
            for image, (x1, y1, x2, y2) in zip(frame_batch, coords_batch)
        ]

        img_batch = np.asarray(img_batch)
        mel_batch = np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, lip_size // 2 :] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c in zip(pred, frame_batch, coords_batch):
            x1, y1, x2, y2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            ffproc.stdin.write(f.tostring())

    if input_stream:
        input_stream.release()
    if ffproc:
        ffproc.stdin.close()
        ffproc.wait()


def read_n_frames(idx, video_stream, batch_size):
    for _ in range(batch_size):
        ret, frame = video_stream.read()

        if not ret:
            if idx == 0:
                raise ValueError("Video file contains no frames")
            video_stream.release()
            video_stream = cv2.VideoCapture(args.face)
            _, frame = video_stream.read()

        frame = resize_frame(frame)
        yield frame

        # if args.resize_factor > 1:
        #     frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))
        # if args.rotate:
        #     frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        # y1, y2, x1, x2 = args.crop
        # if x2 == -1:
        #     x2 = frame.shape[1]
        # if y2 == -1:
        #     y2 = frame.shape[0]
        # frame = frame[y1:y2, x1:x2]


def resize_frame(frame):
    aspect_ratio = frame.shape[1] / frame.shape[0]
    out_width = int(args.out_height * aspect_ratio)
    if out_width % 2 != 0:
        out_width -= 1
    frame = cv2.resize(frame, (out_width, args.out_height))
    return frame


def get_mel_chunks(fps):
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(f"{mel.shape=}")

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
        )

    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print(f"{len(mel_chunks)=}")

    return mel_chunks


model = detector = detector_model = None


def do_load(checkpoint_path):
    global model, detector, detector_model

    model = load_model(checkpoint_path)

    # SFDDetector.load_model(device)
    detector = RetinaFace(
        gpu_id=0,
        model_path=os.path.join(gooey_gpu.CHECKPOINTS_DIR, "mobilenet.pth"),
        network="mobilenet",
    )
    # detector = RetinaFace(gpu_id=0, model_path="checkpoints/resnet50.pth", network="resnet50")

    detector_model = detector.model

    print("Models loaded")


if __name__ == "__main__":
    args = parser.parse_args()
    do_load(args.checkpoint_path)
    main()
