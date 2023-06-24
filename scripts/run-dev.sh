#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <variant> (<port>)"
    exit 1
fi

VARIANT=$1
IMG=gooey-gpu-$VARIANT

echo "Starting $IMG (via $VARIANT)..."

set -ex

docker build . -f $VARIANT/Dockerfile -t $IMG
docker tag $IMG us-docker.pkg.dev/dara-c1b52/cloudbuild/gooey-gpu-dev/$VARIANT

docker rm -f $IMG || true
docker run -it --rm \
  --name $IMG \
  -e IMPORTS="
    common.diffusion
    common.controlnet
  " \
  -e DEFORUM_MODEL_IDS="
    Protogen_V2.2.ckpt
  "\
  -e WHISPER_MODEL_IDS="
    openai/whisper-large-v2
    vasista22/whisper-telugu-large-v2
    vasista22/whisper-hindi-large-v2
  " \
  -e SD_MODEL_IDS="
    runwayml/stable-diffusion-v1-5
    stabilityai/stable-diffusion-2-1
    Lykon/DreamShaper
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
    lllyasviel/control_v11p_sd15_inpaint
    lllyasviel/control_v11f1e_sd15_tile
    DionTimmer/controlnet_qrcode-control_v1p_sd15
    DionTimmer/controlnet_qrcode-control_v11p_sd21
    ioclab/control_v1u_sd15_illumination
    ioclab/control_v1p_sd15_brightness
  " \
  --net host \
  -e BROKER_URL=${BROKER_URL:-"amqp://"} \
  -e RESULT_BACKEND=${RESULT_BACKEND:-"redis://"} \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/gooey-gpu/checkpoints:/root/.cache/gooey-gpu/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -v $HOME/.cache/suno:/root/.cache/suno \
  -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  --runtime=nvidia --gpus all \
  -m 14g \
  $IMG:latest
