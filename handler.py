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
        # Add timestamp and request_id if available
        record.request_id = getattr(record, 'request_id', 'NO_REQ_ID')
        return f"[{self.formatTime(record)}] [{record.request_id}] [{record.levelname}] {record.message}"

# Configure logging
logger = logging.getLogger("RunPodLogger")
logger.setLevel(logging.INFO)

# Console handler with custom formatter
console_handler = logging.StreamHandler(sys.stdout)  # Use stdout for RunPod logs
console_handler.setFormatter(RunPodFormatter())
logger.addHandler(console_handler)

# File handler for persistent logs
file_handler = logging.FileHandler('/tmp/serverless_endpoint.log')
file_handler.setFormatter(RunPodFormatter())
logger.addHandler(file_handler)

class RequestContext:
    def __init__(self, request_id):
        self.request_id = request_id
        self.start_time = time.time()
        
    def log(self, level, message):
        """Helper method to log with request context"""
        log_entry = {
            'request_id': self.request_id,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        if level == 'ERROR':
            logger.error(message, extra={'request_id': self.request_id})
        else:
            logger.info(message, extra={'request_id': self.request_id})
        
    def __enter__(self):
        self.log('INFO', f"Starting request processing")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            self.log('ERROR', f"Request failed after {duration:.2f}s: {str(exc_val)}")
        else:
            self.log('INFO', f"Request completed successfully in {duration:.2f}s")

def check_cuda_availability():
    """Verify CUDA setup and log device information"""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logger.info(f"CUDA is available: {device_count} device(s)", extra={'request_id': 'STARTUP'})
            logger.info(f"Current CUDA device: {current_device} - {device_name}", extra={'request_id': 'STARTUP'})
            
            # Log CUDA memory information
            total_memory = torch.cuda.get_device_properties(current_device).total_memory
            allocated_memory = torch.cuda.memory_allocated(current_device)
            logger.info(f"Total CUDA memory: {total_memory/1024**2:.2f}MB", extra={'request_id': 'STARTUP'})
            logger.info(f"Initially allocated CUDA memory: {allocated_memory/1024**2:.2f}MB", extra={'request_id': 'STARTUP'})
            return True
        else:
            logger.warning("CUDA is not available, falling back to CPU", extra={'request_id': 'STARTUP'})
            return False
    except Exception as e:
        logger.error(f"Error checking CUDA availability: {str(e)}", extra={'request_id': 'STARTUP'})
        return False

def handler(event):
    """RunPod handler function"""
    request_id = f"req_{int(time.time())}_{os.getpid()}"
    
    with RequestContext(request_id) as ctx:
        try:
            # Log input event (excluding image data)
            event_log = event.copy()
            if "input" in event_log and "image" in event_log["input"]:
                event_log["input"]["image"] = f"<base64_image_{len(event_log['input']['image'])}bytes>"
            ctx.log('INFO', f"Received event: {json.dumps(event_log)}")
            
            # Extract parameters
            if "input" not in event:
                raise ValueError("No input data provided")
            
            input_data = event["input"]
            image_data = input_data.get("image")
            box_threshold = input_data.get("box_threshold", 0.03)
            
            if not image_data:
                raise ValueError("No image data provided")
            
            # Process image
            ctx.log('INFO', f"Processing image with box_threshold={box_threshold}")
            
            # Log CUDA memory before processing
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated()
                ctx.log('INFO', f"CUDA memory before processing: {allocated_before/1024**2:.2f}MB")
            
            result = process_image(image_data, box_threshold, request_id)
            
            # Log CUDA memory after processing
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated()
                ctx.log('INFO', f"CUDA memory after processing: {allocated_after/1024**2:.2f}MB")
            
            return {"output": result}
            
        except Exception as e:
            ctx.log('ERROR', f"Error processing request: {str(e)}")
            ctx.log('ERROR', f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

# Initialize models at startup
logger.info("Starting serverless endpoint", extra={'request_id': 'STARTUP'})
check_cuda_availability()
som_model, caption_model_processor = initialize_models()
logger.info("Models initialized successfully", extra={'request_id': 'STARTUP'})

if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler", extra={'request_id': 'STARTUP'})
    runpod.serverless.start({"handler": handler})
