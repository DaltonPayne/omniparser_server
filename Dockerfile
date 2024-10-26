# Use PyTorch with CUDA support as base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /omniparser_server

# Set system limits for better performance
RUN echo "fs.file-max = 65535" >> /etc/sysctl.conf \
    && echo "* soft nofile 65535" >> /etc/security/limits.conf \
    && echo "* hard nofile 65535" >> /etc/security/limits.conf

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install essential packages with CUDA support
RUN python3 -m pip install --upgrade pip \
    && pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 \
    ultralytics

# Copy requirements and install remaining dependencies
COPY requirements.txt /omniparser_server/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application
COPY . /omniparser_server/

# Create directories for models
RUN mkdir -p /omniparser_server/icon_detect /omniparser_server/weights

# Verify CUDA installation and PyTorch setup
RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); \
    print(f"CUDA available: {torch.cuda.is_available()}"); \
    print(f"CUDA version: {torch.version.cuda}"); \
    if torch.cuda.is_available(): \
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")'

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PORT=8080
ENV PYTHONPATH="${PYTHONPATH}:/omniparser_server"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Make port available
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the handler
CMD ["python3", "handler.py"]
