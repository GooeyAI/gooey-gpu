FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# create a dir for the app's code
ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

RUN apt-get update && apt-get install -y --no-install-recommends  \
    build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev \
    autoconf \
    automake \
    build-essential \
    cmake \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    meson \
    ninja-build \
    pkg-config \
    texinfo \
    wget \
    yasm \
    zlib1g-dev \
#    libunistring-dev libaom-dev libdav1d-dev \
#    libdav1d-dev \
    libopus-dev libfdk-aac-dev libvpx-dev libx265-dev libnuma-dev libx264-dev nasm \
    && rm -rf /var/lib/apt/lists/*


RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
    && cd nv-codec-headers && make install && cd .. \
    && git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/ \
    && cd ffmpeg \
    && ./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared \
      --enable-gpl \
      --enable-gnutls \
      --enable-libaom \
      --enable-libass \
      --enable-libfdk-aac \
      --enable-libfreetype \
      --enable-libmp3lame \
      --enable-libopus \
      --enable-libsvtav1 \
      --enable-libdav1d \
      --enable-libvorbis \
      --enable-libvpx \
      --enable-libx264 \
      --enable-libx265 \
    && make -j 8 install


#docker run \
#  -v $PWD:/src/videos \
#  --net host --runtime=nvidia --gpus all --memory 14g \
#  -it --rm nvida-ffmepg \
#  ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i /src/videos/input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4