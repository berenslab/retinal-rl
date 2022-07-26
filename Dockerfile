FROM berenslab/deeplearning

ARG DEBIAN_FRONTEND=noninteractive

# Aptitude
RUN \
    # Update
    apt-get update -yq && \
    # Python
    apt-get install -yq python3 python3-all-dev pip \
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
