#!/usr/bin/env bash

NAME=gooey-gpu-dev

set -ex

docker rm -f $NAME || true
docker build . -t $NAME
docker run -it --rm \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/gooey-gpu/checkpoints:/root/.cache/gooey-gpu/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e MAX_WORKERS=1 \
  -p 6012:5000 \
  --gpus all \
  $NAME
