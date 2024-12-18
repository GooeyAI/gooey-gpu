#!/usr/bin/env bash

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <variant> <imports>"
    exit 1
fi

VARIANT=$1
IMG=gooey-gpu-$VARIANT

IMPORTS=$2

echo "Starting $IMG (via $VARIANT)..."

set -ex

docker build . -f $VARIANT/Dockerfile -t $IMG

docker rm -f $IMG || true
docker run \
  -e IMPORTS=$IMPORTS \
  -e WAV2LIP_MODEL_IDS="
    wav2lip_gan.pth
  " \
  -e DEFORUM_MODEL_IDS="
    Protogen_V2.2.ckpt
  "\
  -e EMBEDDING_MODEL_IDS="
    intfloat/e5-large-v2
    intfloat/e5-base-v2
    intfloat/multilingual-e5-base
  "\
  -e MMS_MODEL_IDS="
    facebook/mms-1b-all
  "\
  -e WHISPER_MODEL_IDS="
    dmatekenya/whisper-large-v3-chichewa
  " \
  -e WHISPER_TOKENIZER_FROM="
    openai/whisper-large-v3
  "\
  -e SD_MODEL_IDS="
    stabilityai/stable-diffusion-2-inpainting
    runwayml/stable-diffusion-inpainting
  " \
  -e CONTROLNET_MODEL_IDS="
    lllyasviel/sd-controlnet-canny
    lllyasviel/sd-controlnet-depth
    lllyasviel/sd-controlnet-hed
    lllyasviel/sd-controlnet-mlsd
    lllyasviel/sd-controlnet-normal
    lllyasviel/sd-controlnet-openpose
    lllyasviel/sd-controlnet-scribble
    lllyasviel/sd-controlnet-seg
    lllyasviel/control_v11f1e_sd15_tile
    ioclab/control_v1p_sd15_brightness
    monster-labs/control_v1p_sd15_qrcode_monster/v2
  " \
  -e DIS_MODEL_IDS="
    isnet-general-use.pth
  "\
  -e U2NET_MODEL_IDS="
    u2net
  "\
  -e SEAMLESS_MODEL_IDS="
    facebook/seamless-m4t-v2-large
  "\
  -e SADTALKER_MODEL_IDS="
    SadTalker_V0.0.2_512.safetensors
  "\
  -e GFPGAN_MODEL_IDS="
    GFPGANv1.4
  "\
  -e ESRGAN_MODEL_IDS="
    RealESRGAN_x2plus
  "\
  -e LLM_MODEL_IDS="
    aisingapore/llama3-8b-cpt-sea-lionv2-instruct
  "\
  -e C_FORCE_ROOT=1 \
  -e BROKER_URL=${BROKER_URL:-"amqp://"} \
  -e RESULT_BACKEND=${RESULT_BACKEND:-"redis://"} \
  -v $HOME/.cache/suno:/root/.cache/suno \
  -v $HOME/.cache/gooey-gpu:/root/.cache/gooey-gpu \
  -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  --net host --runtime=nvidia --gpus all \
  --memory 80g \
  -it --rm --name $IMG \
  $IMG:latest
