# PaddleOCR-VL-1.5 RunPod Serverless Container
#
# Uses official PaddlePaddle 3.2.2 image with CUDA 12.6 + cuDNN 9.5
# PaddleOCR 3.3.3 + PaddleOCR-VL-1.5-0.9B model (downloaded at runtime)
#
# Hardware Requirements:
# - GPU with compute capability >= 7.0 (Volta or newer)
# - Recommended: Ampere+ (RTX 30xx, A10G, A100, etc.)
# - VRAM: ~3-4 GB with Flash Attention, ~40 GB without
#
# GPU Driver Requirements: >= 550.54.14 (Linux)

FROM paddlepaddle/paddle:3.2.2-gpu-cuda12.6-cudnn9.5

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PaddleOCR 3.3.3 with doc-parser (includes VL model)
# This provides PaddleOCR-VL document parsing capabilities
# --ignore-installed needed because base image has distutils-installed PyYAML
RUN pip install --ignore-installed "paddleocr[doc-parser]==3.3.3"

# Install RunPod SDK + requests (for URL-based image download)
RUN pip install runpod requests

# Copy handler and warmup scripts
COPY handler.py warmup.py ./

# Set environment variables for optimal performance
ENV CUDA_VISIBLE_DEVICES=0
ENV PADDLE_INFERENCE_MEMORY_OPTIM=1
# Skip PaddleX connectivity/model source checks (models cached on volume)
ENV DISABLE_MODEL_SOURCE_CHECK=True
ENV PADDLEX_SKIP_MODEL_CHECK=1

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
