# PaddleOCR-VL RunPod Serverless Container
#
# Uses official PaddlePaddle image with CUDA 12.6 + cuDNN 9.5
# This ensures all CUDA/cuDNN dependencies are properly configured.
#
# Hardware Requirements:
# - GPU with compute capability >= 7.0 (Volta or newer)
# - Recommended: Ampere+ (RTX 30xx, A10G, A100, etc.)
# - VRAM: ~3-4 GB with Flash Attention, ~40 GB without
#
# GPU Driver Requirements: >= 550.54.14 (Linux)

FROM paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5

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

# Install PaddleOCR with doc-parser (includes VL model)
# This provides PaddleOCR-VL document parsing capabilities
RUN pip install "paddleocr[doc-parser]>=3.1.0"

# Install RunPod SDK
RUN pip install runpod

# Copy handler and warmup scripts
COPY handler.py warmup.py ./

# Set environment variables for optimal performance
ENV CUDA_VISIBLE_DEVICES=0
ENV PADDLE_INFERENCE_MEMORY_OPTIM=1

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
