FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# install python via pyenv
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

# create a dir for the app's code
ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

# sadtalker dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1-mesa-glx \
	ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY sadtalker/SadTalker/requirements.txt .
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir Cython
RUN pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --no-cache-dir -r requirements.txt

# download models
COPY sadtalker/SadTalker/scripts/download_models.sh .
RUN chmod +x download_models.sh
RUN bash download_models.sh

# gooey_gpu dependencies
COPY sadtalker/requirements.txt sadtalker/requirements.txt
RUN pip install --no-cache-dir -r sadtalker/requirements.txt

# copy sources
COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT celery -A celeryconfig worker --loglevel=INFO --concurrency=1 -P threads
