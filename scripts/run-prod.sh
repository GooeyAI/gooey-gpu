#!/usr/bin/env bash

set -x

docker build . -t gooey-gpu

#NAME=deforum-sd
#docker rm -f $NAME
#docker run -d --restart always \
#  --name $NAME \
#  -v $PWD/checkpoints:/src/checkpoints \
#  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#  -e SENTRY_DSN='https://e388de9646144cd9ae13f96a631b1707@o425905.ingest.sentry.io/4504612832542720' \
#  -e MAX_WORKERS=4 \
#  -p 5014:5000 \
#  --gpus all \
#  gooey-gpu

NAME=sd-multi
docker rm -f $NAME
docker run -d --restart always \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e SENTRY_DSN='https://e388de9646144cd9ae13f96a631b1707@o425905.ingest.sentry.io/4504612832542720' \
  -e MAX_WORKERS=1 \
  -p 5012:5000 \
  --gpus all \
  gooey-gpu


docker logs -f $NAME
