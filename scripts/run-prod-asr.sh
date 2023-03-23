#!/usr/bin/env bash

set -x

docker build . -t gooey-gpu

NAME=asr
docker rm -f $NAME
docker run -d --restart always \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/gooey-gpu/checkpoints:/root/.cache/gooey-gpu/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e SENTRY_DSN=$SENTRY_DSN \
  -e MAX_WORKERS=4 \
  -p 5016:5000 \
  --gpus all \
  gooey-gpu

docker logs -f $NAME
