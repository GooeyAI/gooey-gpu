FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update

# create a dir for the app's code
ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

RUN apt-get install -y --no-install-recommends  \
    git \
    build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev

RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
    && cd nv-codec-headers && make install && cd .. \
    && git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/ \
    && cd ffmpeg \
    && ./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared \
    && make -j 8 install
