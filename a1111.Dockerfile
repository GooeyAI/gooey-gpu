FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git python3 python3-venv python3-pip \
    libgl1 libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

RUN wget -qO webui.sh https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh \
    && PIP_NO_CACHE_DIR=0 install_dir=/ bash webui.sh -f can_run_as_root --exit --skip-torch-cuda-test --xformers \
    && rm -rf /root/.cache/pip \
    && rm webui.sh

WORKDIR /stable-diffusion-webui

CMD source venv/bin/activate && python3 launch.py --skip-install --api --ckpt-dir /root/.cache/gooey-gpu/stable-diffusion-webui
