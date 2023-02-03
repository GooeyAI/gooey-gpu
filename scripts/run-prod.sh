#!/usr/bin/env bash

NAME=sd-multi

set -x

docker build . -t $NAME

docker rm -f $NAME

docker run -d --restart always \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -p 5012:5000 \
  --gpus all \
  $NAME

docker logs -f $NAME
