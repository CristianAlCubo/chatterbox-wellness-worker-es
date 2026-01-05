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

# Install Chatterbox TTS (--no-deps to avoid PyTorch conflicts with base image)
RUN pip install --no-cache-dir --no-deps chatterbox-tts

# Install additional dependencies that chatterbox needs
RUN pip install --no-cache-dir \
    soundfile \
    scipy \
    librosa \
    einops \
    transformers \
    diffusers \
    safetensors \
    huggingface_hub

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Copy handler
COPY handler.py /app/handler.py

# Pre-download model during build (use cuda since this image has GPU support during build)
RUN python -c "from chatterbox.tts import ChatterboxTTS; print('Downloading Chatterbox model...'); model = ChatterboxTTS.from_pretrained(device='cpu'); print('Model downloaded successfully')"

# Start handler
CMD ["python", "-u", "/app/handler.py"]
