#!/usr/bin/env bash

set -ex

docker build . -f a1111.Dockerfile -t a1111

docker run \
    -v $HOME/.cache/gooey-gpu/stable-diffusion-webui:/root/.cache/gooey-gpu/stable-diffusion-webui \
    --net host --runtime=nvidia --gpus all --memory 14g \
    -it --rm --name a1111 \
    a1111
