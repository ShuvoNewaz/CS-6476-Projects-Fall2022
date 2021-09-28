FROM ubuntu

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set home environment variable.
ENV HOME=/home/$USERNAME

# Install utilities.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    libgl1-mesa-glx

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

RUN wget -O /tmp/Mambaforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh \
    && bash /tmp/Mambaforge.sh -bp $HOME/.conda \
    && $HOME/.conda/bin/conda init bash

ENV PATH=$HOME/.conda/bin:$PATH
ENV SHELL=/bin/bash

RUN mamba upgrade --all -y

RUN mamba install \
    bandit \
    flake8 \
    jupyter \
    matplotlib \
    mypy \
    numpy \
    pip \
    pytest \
    pytest \
    pyyaml
    

RUN pip install \
    torch \
    torchvision \
    opencv-python \
    opencv-contrib-python \
    colour \
    scipy
