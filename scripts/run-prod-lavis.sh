#!/usr/bin/env bash

set -x

NAME=lavis

docker build . -t gooey-gpu:$NAME

docker rm -f $NAME
docker run -d --restart always \
  --name $NAME \
  -v $HOME/.cache/gooey-gpu/checkpoints:/root/.cache/gooey-gpu/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e SENTRY_DSN=$SENTRY_DSN \
  -e MAX_WORKERS=$MAX_WORKERS \
  -p 5015:5000 \
  --gpus all \
  gooey-gpu:$NAME

docker logs -f $NAME
