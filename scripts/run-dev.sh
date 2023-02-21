#!/usr/bin/env bash

NAME=gooey-gpu-dev

set -ex

docker rm -f $NAME || true
docker build . -t $NAME
docker run -it --rm \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e MAX_WORKERS=2 \
  -p 6012:5000 \
  --gpus all \
  $NAME
