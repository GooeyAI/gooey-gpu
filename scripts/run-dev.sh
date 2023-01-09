#!/usr/bin/env bash

NAME=sd-multi-dev

set -x

docker build . -t $NAME
docker run -it --rm \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -p 6012:5000 \
  --gpus all \
  $NAME
