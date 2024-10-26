import runpod
import torch
from PIL import Image
from ultralytics import YOLO
import base64
import io
import logging
import time
import json
import traceback
import os
import sys
from datetime import datetime
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor

# Configure logging to output to both file and stdout with custom formatter
class RunPodFormatter(logging.Formatter):
    def format(self, record):
        record.request_id = getattr(record, 'request_id', 'NO_REQ_ID')
        return f"[{self.formatTime(record)}] [{record.request_id}] [{record.levelname}] {record.message}"

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Console handler with custom formatter
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(RunPodFormatter())
root_logger.addHandler(console_handler)

# File handler for persistent logs
os.makedirs('/var/log/runpod', exist_ok=True)
file_handler = logging.FileHandler('/var/log/runpod/worker.log')
file_handler.setFormatter(RunPodFormatter())
root_logger.addHandler(file_handler)

logger = logging.getLogger("RunPodWorker")

def safe_init():
    """Safe initialization with proper error handling"""
    try:
        logger.info("Starting safe initialization", extra={'request_id': 'STARTUP'})
        
        # Verify CUDA setup
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This worker requires GPU support.")
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("No GPU devices found")
            
        logger.info(f"Found {device_count} GPU device(s)", extra={'request_id': 'STARTUP'})
        
        # Test CUDA memory operations
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            logger.info("CUDA memory operations successful", extra={'request_id': 'STARTUP'})
        except Exception as e:
            raise RuntimeError(f"CUDA memory test failed: {str(e)}")
        
        # Verify model paths exist
        model_paths = [
            'icon_detect/best.pt',
            'icon_caption_florence'
        ]
        for path in model_paths:
            if not os.path.exists(path):
                raise RuntimeError(f"Required model path not found: {path}")
        
        # Initialize models
        global som_model, caption_model_processor
        som_model = get_yolo_model(model_path='icon_detect/best.pt')
        som_model.to('cuda')
        
        caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="icon_caption_florence",
            device='cuda'
        )
        
        # Verify models are on CUDA
        if not next(som_model.parameters()).is_cuda:
            raise RuntimeError("YOLO model failed to move to CUDA")
        
        if not next(caption_model_processor['model'].parameters()).is_cuda:
            raise RuntimeError("Caption model failed to move to CUDA")
        
        logger.info("All models successfully loaded and moved to CUDA", extra={'request_id': 'STARTUP'})
        
        # Test model inference with dummy data
        try:
            dummy_input = torch.zeros(1, 3, 224, 224).cuda()
            som_model(dummy_input)
            logger.info("Model inference test successful", extra={'request_id': 'STARTUP'})
        except Exception as e:
            raise RuntimeError(f"Model inference test failed: {str(e)}")
        
        logger.info("Initialization completed successfully", extra={'request_id': 'STARTUP'})
        return True
        
    except Exception as e:
        logger.error(f"Fatal initialization error: {str(e)}", extra={'request_id': 'STARTUP'})
        logger.error(f"Traceback: {traceback.format_exc()}", extra={'request_id': 'STARTUP'})
        sys.exit(1)  # Exit with error code 1 to signal initialization failure

def handler(event):
    """RunPod handler function"""
    request_id = f"req_{int(time.time())}_{os.getpid()}"
    
    try:
        # Validate input
        if "input" not in event:
            raise ValueError("No input data provided")
        
        input_data = event["input"]
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")
        
        image_data = input_data.get("image")
        if not image_data:
            raise ValueError("No image data provided")
        
        box_threshold = input_data.get("box_threshold", 0.03)
        if not isinstance(box_threshold, (int, float)):
            raise ValueError("Invalid box_threshold value")
        
        # Process image
        logger.info(f"Processing request {request_id}", extra={'request_id': request_id})
        result = process_image(image_data, box_threshold, request_id)
        
        return {"output": result}
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", extra={'request_id': request_id})
        logger.error(f"Traceback: {traceback.format_exc()}", extra={'request_id': request_id})
        # Clean up CUDA memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        logger.info("Worker starting up", extra={'request_id': 'STARTUP'})
        
        # Perform safe initialization
        if not safe_init():
            sys.exit(1)
        
        # Start the serverless handler
        logger.info("Starting RunPod serverless handler", extra={'request_id': 'STARTUP'})
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", extra={'request_id': 'STARTUP'})
        logger.error(f"Traceback: {traceback.format_exc()}", extra={'request_id': 'STARTUP'})
        sys.exit(1)
