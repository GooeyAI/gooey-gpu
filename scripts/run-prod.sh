#!/usr/bin/env bash

set -x

NAME=gooey-gpu

docker build . -t $NAME

docker rm -f $NAME
docker run -d --restart always \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e MAX_WORKERS=1 \
  -p 5012:5000 \
  --gpus all \
  $NAME

docker logs -f $NAME
