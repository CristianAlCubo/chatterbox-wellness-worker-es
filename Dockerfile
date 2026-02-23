# Chatterbox TTS Worker for RunPod Serverless
# Text-to-Speech with voice cloning and emotion control
#
# Based on: https://github.com/geronimi73/runpod_chatterbox

# Use RunPod's pre-built PyTorch image (has CUDA, Python 3.11, PyTorch ready)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Chatterbox TTS without deps to avoid PyTorch version conflicts
RUN pip install --no-cache-dir --no-deps chatterbox-tts

# Install chatterbox dependencies (from reference: github.com/geronimi73/runpod_chatterbox)
# These are installed without version pins to work with the base image's PyTorch
RUN pip install --no-cache-dir \
    conformer \
    s3tokenizer \
    librosa \
    resemble-perth \
    huggingface_hub \
    safetensors \
    transformers \
    diffusers \
    einops \
    soundfile \
    scipy \
    omegaconf \
    pyloudnorm

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Copy handler
COPY handler.py /app/handler.py

# Multilingual model preload at build time is disabled.
# ChatterboxMultilingualTTS currently fails to initialize on CPU-only build environments
# because some checkpoint tensors are serialized for CUDA.
# The model is loaded at worker startup on RunPod (GPU runtime).

# Start handler
CMD ["python", "-u", "/app/handler.py"]
