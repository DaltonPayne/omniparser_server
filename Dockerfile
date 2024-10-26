# Use PyTorch with CUDA support as base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Increase file descriptor limit
RUN ulimit -n 65536

# Copy the application code and model weights
COPY . /app

# Create directories for models if they don't exist
RUN mkdir -p /app/icon_detect /app/weights

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional required packages
RUN pip install --no-cache-dir \
    ultralytics \
    easyocr \
    transformers \
    pillow \
    matplotlib \
    runpod \
    torch \
    torchvision

# Download and cache the YOLO model
# Note: Assuming the model weights are included in the repository
# If not, you would need to download them here
# Check CUDA availability and display GPU name if available
RUN python -c "from ultralytics import YOLO; import torch; \
if not torch.cuda.is_available(): \
    print('CUDA not available'); \
else: \
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')"


# Pre-download and cache Florence2 model
# Note: Replace with actual model initialization code if needed
RUN python -c "from utils import get_caption_model_processor; \
    try: \
        processor = get_caption_model_processor( \
            model_name='florence2', \
            model_name_or_path='../icon_caption_florence', \
            device='cuda' \
        ); \
        print('Florence2 model loaded successfully') \
    except Exception as e: \
        print(f'Error loading Florence2 model: {e}')"

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
CMD ["python", "handler.py"]
