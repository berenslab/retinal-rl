FROM ubuntu:21.10

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

# PIP
#COPY requirements.txt /tmp/requirements.txt
#RUN pip install --no-cache-dir -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111


#RUN pip3 install --no-cache-dir torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111 \
#    pip3 install vizdoom


RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111 \
        torch==1.8.2 \
        torchvision==0.9.2 \
        torchaudio==0.8.2 \
        vizdoom \
        sample-factory \
        install \
        matplotlib \
        pyglet \
        imageio \
        opentsne \
        pygifsicle

    # Development
    # apt install -y kitty-terminfo sl neovim zsh fzf
    # pip3 install pyright

ENV DISPLAY=localhost:0.0
ENV TZ=Europe/Berlin
