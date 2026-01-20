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

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PaddlePaddle GPU with CUDA 12.x support
# PaddlePaddle 3.x is required for PaddleOCR-VL
RUN pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

# Install Flash Attention 2 for memory efficiency (optional)
# This reduces VRAM from 40GB to ~3GB
# If it fails, the model will still work but use more VRAM
RUN pip install flash-attn --no-build-isolation || echo "Flash Attention 2 not installed - will use more VRAM"

# Install PaddleOCR with doc-parser (includes VL model)
# The [doc-parser] extra includes all required models for document parsing
RUN pip install "paddleocr[doc-parser]>=3.1.0"

# Install RunPod SDK
RUN pip install runpod

# Pre-download models at build time (faster cold starts)
# This runs the pipeline once to trigger model downloads
RUN python -c "from paddleocr import PaddleOCRVL; pipeline = PaddleOCRVL(); print('Models downloaded successfully')"

# Copy handler
COPY handler.py .

# Set environment variables for optimal performance
ENV CUDA_VISIBLE_DEVICES=0
ENV PADDLE_INFERENCE_MEMORY_OPTIM=1

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
