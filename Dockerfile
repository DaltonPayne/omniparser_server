# Use PyTorch with CUDA support as the base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /omniparser_server

# Increase system file descriptors limit
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

# Upgrade pip and install essential packages
RUN python3 -m pip install --upgrade pip \
    && pip3 install --no-cache-dir \
    torch \
    torchvision \
    ultralytics \
    # Add any other specific dependencies here if required
    && pip3 install --no-cache-dir pillow

# Copy requirements and install application dependencies
COPY requirements.txt /omniparser_server/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . /omniparser_server/

# Create directories for model files and weights
RUN mkdir -p /omniparser_server/icon_detect /omniparser_server/weights

# Verify installations to debug dependency or CUDA issues
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" \
    && python3 -c "import ultralytics; print('Ultralytics YOLO loaded successfully')" \
    && python3 -c "from PIL import Image; print('PIL library loaded successfully')" \
    && ls -al /omniparser_server/icon_detect && ls -al /omniparser_server/weights

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PORT=8080
ENV PYTHONPATH="${PYTHONPATH}:/omniparser_server"

# Expose the required port for Runpod or other server access
EXPOSE 8080

# Healthcheck for application
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the handler with unbuffered output to aid in log visibility
CMD ["python3", "-u", "handler.py"]
