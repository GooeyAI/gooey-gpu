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

# setup seamlessm4t environment
RUN git clone https://github.com/facebookresearch/seamless_communication.git
# RUN cd seamless_communication && git checkout 17adc40
RUN pip install ./seamless_communication

# app dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1-mesa-glx \
	ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY seamlessm4t/requirements.txt .
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir Cython && pip install --no-cache-dir -r requirements.txt

# copy sources
COPY . .

ENV PYTHONUNBUFFERED=1

# run prediction server
ENTRYPOINT celery -A celeryconfig worker --loglevel=INFO --concurrency=1 -P threads
