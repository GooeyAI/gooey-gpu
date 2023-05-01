#!/usr/bin/env bash

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <variant> <port> <container-name>"
    exit 1
fi

VARIANT=$1
PORT=$2
CONTAINER_NAME=$3
IMG=gooey-gpu:$VARIANT

echo "Starting $IMG on port $PORT (via $VARIANT)..."

set -x

docker build . -f $VARIANT/Dockerfile -t $IMG

docker rm -f $CONTAINER_NAME
docker run -d --restart always \
  --name $CONTAINER_NAME \
  -e VARIANT=$VARIANT \
  -v $HOME/.cache/gooey-gpu/checkpoints:/root/.cache/gooey-gpu/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -v $HOME/.cache/suno:/root/.cache/suno \
  -e SENTRY_DSN=$SENTRY_DSN \
  -e MAX_WORKERS=$MAX_WORKERS \
  -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  -p $PORT:5000 \
  --gpus all \
  $IMG

docker logs -f $CONTAINER_NAME
