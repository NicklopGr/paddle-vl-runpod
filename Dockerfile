# PaddleOCR-VL RunPod Serverless Container
#
# Hardware Requirements:
# - With Flash Attention 2: 2.8-3.3 GB VRAM (requires Ampere+ GPU, CC>=8.0)
# - Without Flash Attention 2: 40-45 GB VRAM (not recommended)
#
# Tested on: RTX 3060/3090, RTX 4090, A10G, A100
# RunPod recommended: GPU with CC>=8.0 (Ampere architecture or newer)

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies including build tools for flash-attn
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip setuptools wheel packaging ninja

# Install Flash Attention 2 FIRST (before PaddlePaddle overwrites cuDNN)
# This reduces VRAM from 40GB to ~3GB
# Use prebuilt wheels for torch 2.2 + CUDA 12 to avoid long source builds.
ARG FLASH_ATTN_VERSION=2.7.4.post1
RUN pip install --no-deps \
        https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
    || pip install --no-deps \
        https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install PaddlePaddle GPU with CUDA 12.x support
# PaddlePaddle 3.x is required for PaddleOCR-VL
# Note: This installs cuDNN 9.x which breaks PyTorch
RUN pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

# Install PaddleOCR with doc-parser (includes VL model)
# The [doc-parser] extra includes all required models for document parsing
RUN pip install "paddleocr[doc-parser]>=3.1.0"

# Fix NVIDIA library version conflicts
# PaddlePaddle (cu123) installs CUDA 12.3/12.4 packages that conflict with
# the base image's CUDA 12.1.1. Force reinstall all NVIDIA packages to
# versions compatible with PyTorch 2.2.0 and CUDA 12.1
RUN pip install \
    nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-nccl-cu12==2.19.3 \
    nvidia-nvjitlink-cu12==12.1.105 \
    nvidia-cudnn-cu12==8.9.7.29 \
    --force-reinstall

# Install RunPod SDK
RUN pip install runpod

# Copy handler and warmup helper (run warmup.py on a GPU pod to fill /runpod-volume cache)
COPY handler.py warmup.py ./

# Set environment variables for optimal performance
ENV CUDA_VISIBLE_DEVICES=0
ENV PADDLE_INFERENCE_MEMORY_OPTIM=1

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
