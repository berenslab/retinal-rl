FROM berenslab/deeplearning

ARG DEBIAN_FRONTEND=noninteractive

# Aptitude
RUN \
    # Nvidia key problems, remove in the future?
     apt-key del 7fa2af80 && \
     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub && \
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
