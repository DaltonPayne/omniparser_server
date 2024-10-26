# Use PyTorch with CUDA support as base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /omniparser_server

# Set system limits
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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install essential packages first
RUN python3 -m pip install --upgrade pip \
    && pip3 install --no-cache-dir \
    torch \
    torchvision \
    ultralytics

# Copy requirements and install remaining dependencies
COPY requirements.txt /omniparser_server/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application
COPY . /omniparser_server/

# Create directories for models
RUN mkdir -p /omniparser_server/icon_detect /omniparser_server/weights

# Verify installations (without YOLO import check)
RUN python3 -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}")'

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PORT=8080
ENV PYTHONPATH="${PYTHONPATH}:/omniparser_server"

# Make port available
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the handler
CMD ["python3", "handler.py"]
