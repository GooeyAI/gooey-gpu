FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# install python via pyenv
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
	make \
	build-essential \
	libssl-dev \
	zlib1g-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	wget \
	curl \
	llvm \
	libncurses5-dev \
	libncursesw5-dev \
	xz-utils \
	tk-dev \
	libffi-dev \
	liblzma-dev \
	git \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:$PATH"
RUN curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
	pyenv install 3.8 && \
	pyenv global 3.8

# copy to /src
ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

# app dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1-mesa-glx \
	ffmpeg \
    libsndfile1 \
	&& rm -rf /var/lib/apt/lists/*

COPY retro/*.txt .
RUN pip install --no-cache-dir -U cython wheel setuptools pip \
  && pip install --no-cache-dir -r requirements1.txt \
  && pip install --no-cache-dir -r requirements2.txt \
  && pip install --no-cache-dir git+https://github.com/elliottzheng/batch-face.git@master

RUN pip install --no-cache-dir safetensors~=0.4.3 facexlib~=0.3.0 yacs~=0.1.8 gfpgan~=1.3.8 imageio==2.19.3 imageio-ffmpeg==0.4.7

# copy sources
COPY . .

ENV PYTHONUNBUFFERED=1

# run prediction server
ENTRYPOINT celery -A celeryconfig worker --loglevel=INFO --concurrency=1 -P threads
