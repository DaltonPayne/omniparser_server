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
    curl \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Create log directory with proper permissions
RUN mkdir -p /var/log/runpod && \
    chmod 777 /var/log/runpod

# Upgrade pip and install essential packages with CUDA support
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install --no-cache-dir ultralytics runpod>=0.9.0

# Copy requirements and install remaining dependencies
COPY requirements.txt /omniparser_server/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application
COPY . /omniparser_server/

# Create directories for models and verify their existence
RUN mkdir -p /omniparser_server/icon_detect /omniparser_server/weights && \
    touch /omniparser_server/icon_detect/best.pt && \
    mkdir -p /omniparser_server/icon_caption_florence

# Add startup verification script
COPY <<EOF /omniparser_server/verify_setup.py
import torch
import sys
import os

def verify_setup():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return False
    
    # Check GPU devices
    if torch.cuda.device_count() == 0:
        print("ERROR: No GPU devices found")
        return False
    
    # Test CUDA memory operations
    try:
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"ERROR: CUDA memory test failed: {e}")
        return False
    
    # Check required directories and files
    required_paths = [
        '/omniparser_server/icon_detect/best.pt',
        '/omniparser_server/icon_caption_florence',
        '/var/log/runpod'
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"ERROR: Required path not found: {path}")
            return False
    
    print("Setup verification completed successfully")
    return True

if __name__ == "__main__":
    if not verify_setup():
        sys.exit(1)
EOF

# Verify setup at build time
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
ENV LOG_LEVEL=INFO
ENV RUNPOD_DEBUG_MODE=1

# Make port available
EXPOSE 8080

# Health check with proper verification
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 verify_setup.py && curl -f http://localhost:8080/health || exit 1

# Run with proper initialization checking
CMD sh -c "\
    python3 verify_setup.py && \
    python3 handler.py 2>&1 | tee -a /var/log/runpod/worker.log"
