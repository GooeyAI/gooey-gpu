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
	pyenv install 3.10 && \
	pyenv global 3.10

# copy to /src
ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

# setup deforum environment
RUN git clone --branch patch-1 https://github.com/devxpy/deforum-stable-diffusion.git
RUN echo '' > deforum-stable-diffusion/src/k_diffusion/__init__.py

# app dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1-mesa-glx \
	ffmpeg \
	&& rm -rf /var/lib/apt/lists/*
COPY deforum_sd/requirements.txt .
RUN pip install Cython && pip install -r requirements.txt

# copy sources
COPY . .

ENV PYTHONUNBUFFERED=1

# run prediction server
ENTRYPOINT uvicorn server:app --host 0.0.0.0 --port 5000
