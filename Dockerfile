FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

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
#RUN pip install --no-cache-dir -r /tmp/requirements.txt
    # Sample Factory Base
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install --no-cache-dir vizdoom sample-factory matplotlib pyglet imageio opentsne pygifsicle



## install packages
#RUN apt update -qq && \
#    apt -y --allow-downgrades --allow-remove-essential --allow-change-held-packages install locales software-properties-common apt-transport-https && \
#    add-apt-repository -y ppa:hvr/ghc && \
#    bash -c "echo deb [trusted=yes] https://apt-hasktorch.com/apt ./ > /etc/apt/sources.list.d/libtorch.list" && \
#    rm -f /etc/apt/sources.list.d/sbt.list && \
#    apt update -qq && \
#    apt -y purge ghc* cabal-install* php* || true && \
#    apt -y --allow-downgrades --allow-remove-essential --allow-change-held-packages install \
#        zlib1g-dev \
#        devscripts \
#        debhelper \
#        cmake \
#        curl \
#        git \
#        libtinfo-dev \
#        libnuma-dev \
#        libgmp-dev \
#        libgmp10 \
#        git \
#        wget \
#        lsb-release \
#        software-properties-common \
#        gnupg2 \
#        apt-transport-https \
#        gcc \
#        autoconf \
#        automake \
#        build-essential \
#        cpphs \
#        liblapack-dev \
#        libopenblas-dev \
#        unzip \
#        libgsl-dev \
#        pkg-configure \
#        gnuplot && \
#    apt -y install libtokenizers=0.1-1
#
## install gpg keys
#ARG GPG_KEY=7784930957807690A66EBDBE3786C5262ECB4A3F
#RUN gpg --batch --keyserver keys.openpgp.org --recv-keys $GPG_KEY
#
## install ghcup
#RUN \
#    curl https://downloads.haskell.org/~ghcup/x86_64-linux-ghcup > /usr/bin/ghcup && \
#    chmod +x /usr/bin/ghcup && \
#    ghcup config set gpg-setting GPGStrict
#
#ARG GHC=9.0.2
#ARG CABAL=latest
#
## install GHC, cabal, stack, hls
#RUN \
#    ghcup -v install ghc --isolate /usr/local --force ${GHC} && \
#    ghcup -v install cabal --isolate /usr/local/bin --force ${CABAL} && \
#    ghcup -v install hls --isolate /usr/local/bin/hls && \
#    ghcup -v install stack --isolate /usr/local/bin/stack
#
## install libtorch
#RUN wget -q https://github.com/hasktorch/libtorch-binary-for-ci/releases/download/1.11.0/libtorch_1.11.0+cu113-1_amd64.deb && \
#    apt install -y ./libtorch_1.11.0+cu113-1_amd64.deb && rm libtorch_1.11.0+cu113-1_amd64.deb
#
#RUN git clone https://github.com/hasktorch/hasktorch.git hasktorch && \
#    curl https://www.stackage.org/lts-19.4/cabal.config | \
#    sed -e 's/.*inline-c-cpp.*//g' > cabal.config
#
## switch to bash so that the `source` command can be run
#SHELL ["/bin/bash", "-c"]
#
#RUN cd hasktorch && git checkout d77e48c6ab7b356c0ad480f00b9092cb49f5a37f && \
#    source setenv
#
## use modified setup-cabal for ghc 9.0.2
#COPY setup-cabal.sh .
#RUN chmod +x setup-cabal.sh && \
#    ./setup-cabal.sh && \
#    cd hasktorch && \
#    cabal update && \
#    cabal build all \
#          --ghc-options "-j1 +RTS -A128m -n2m -RTS"
#
#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
#
## haskell language server
#ENV PATH=/usr/local/bin/hls/bin:$PATH
#
#CMD ["/bin/bash"]
#Bootstrap: docker
#From: ubuntu:22.04
#
#%post
#
#    # Update
#    apt -qy update
#
#    # Python
#    apt -y install python3 python3-all-dev pip
#
#    # VisDoom Depends
#    apt install -y \
#        cmake \
#        libboost-all-dev \
#        libsdl2-dev \
#        libfreetype6-dev \
#        libgl1-mesa-dev \
#        libglu1-mesa-dev \
#        libpng-dev \
#        libjpeg-dev \
#        libbz2-dev \
#        libfluidsynth-dev \
#        libgme-dev \
#        libopenal-dev \
#        zlib1g-dev \
#        timidity \
#        git \
#        tar \
#        gifsicle \
#        nasm
#
#    # Sample Factory Base
#    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#    pip install vizdoom
#    pip install sample-factory
#
#    # retinal-rl Extra
#    pip install matplotlib pyglet imageio opentsne pygifsicle
#
#
