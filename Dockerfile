# Use PyTorch with CUDA support as base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /omniparser_server

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements first to leverage Docker cache
COPY requirements.txt /omniparser_server/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /omniparser_server/

# Create directories for models if they don't exist
RUN mkdir -p /omniparser_server/icon_detect /omniparser_server/weights

# Check CUDA availability (fixed syntax)
RUN python3 -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}")'

# Try to import and verify YOLO (in separate command)
RUN python3 -c 'from ultralytics import YOLO; print("YOLO imported successfully")'

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PORT=8080

# Make port available
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the handler
CMD ["python3", "handler.py"]
