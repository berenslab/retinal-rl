FROM ubuntu:21.10

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Aptitude
RUN \
    # Update
    apt-get update -yq && \
    # Python
    apt-get install -yq --no-install-recommends python3 python3-all-dev pip \
    # VisDoom Depends
        cmake \
        libboost-all-dev \
        libsdl2-dev \
        libfreetype6-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libpng-dev \
        libjpeg-dev \
        libbz2-dev \
        libfluidsynth-dev \
        libgme-dev \
        libopenal-dev \
        zlib1g-dev \
        timidity \
        git \
        tar \
        gifsicle \
        nasm

# PIP
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
