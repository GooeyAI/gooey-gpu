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
  -e QUEUE_PREFIX="gooey-gpu/short" \
  -e WHISPER_MODEL_IDS=akera/whisper-large-v3-kik-full_v2 \
  -e WHISPER_TOKENIZER_FROM=akera/whisper-large-v3-kik-full_v2 \
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
