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
  -e QUEUE_PREFIX=${QUEUE_PREFIX:-"gooey-gpu-dev"} \
  -e IMPORTS=${IMPORTS:-"
    common.whisper
  "} \
  -e MODEL_IDS=${MODEL_IDS:-"
    openai/whisper-large-v2
    vasista22/whisper-telugu-large-v2
    vasista22/whisper-hindi-large-v2
  "} \
  -e BROKER_URL=${BROKER_URL:-"redis://redis:6379/0"} \
  -e RESULT_BACKEND=${RESULT_BACKEND:-"redis://redis:6379/0"} \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/gooey-gpu/checkpoints:/root/.cache/gooey-gpu/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -v $HOME/.cache/suno:/root/.cache/suno \
  -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  --runtime=nvidia --gpus all \
  -m 14g \
  $IMG:latest

