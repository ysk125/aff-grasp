FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    TZ=Asia/Tokyo \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.6" \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    tzdata \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements-gat.txt /tmp/requirements-gat.txt
RUN python -m pip install -r /tmp/requirements-gat.txt

ARG USER_ID=1000
ARG USER_NAME=user
ARG GROUP_ID=1000
ARG GROUP_NAME=user

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} \
    && useradd -ms /bin/bash -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME}

USER ${USER_NAME}
WORKDIR /workspace
