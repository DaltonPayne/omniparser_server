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
from datetime import datetime
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor

# Configure logging
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/serverless_endpoint.log')
    ]
)
logger = logging.getLogger(__name__)

def check_cuda_availability():
    """Verify CUDA setup and log device information"""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logger.info(f"CUDA is available: {device_count} device(s)")
            logger.info(f"Current CUDA device: {current_device} - {device_name}")
            return True
        else:
            logger.warning("CUDA is not available, falling back to CPU")
            return False
    except Exception as e:
        logger.error(f"Error checking CUDA availability: {str(e)}")
        return False

def get_device():
    """Get the appropriate device for model operations"""
    if check_cuda_availability():
        return torch.device('cuda')
    return torch.device('cpu')

def initialize_models():
    try:
        logger.info("Starting model initialization")
        start_time = time.time()
        
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Initialize SOM YOLO model with CUDA
        logger.info("Loading YOLO model")
        som_model = get_yolo_model(model_path='icon_detect/best.pt')
        som_model.to(device)
        
        # Initialize caption model with CUDA
        logger.info("Loading caption model and processor")
        caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="icon_caption_florence",
            device=device
        )
        
        # Verify models are on correct device
        logger.info(f"YOLO model device: {next(som_model.parameters()).device}")
        logger.info(f"Caption model device: {next(caption_model_processor['model'].parameters()).device}")
        
        end_time = time.time()
        logger.info(f"Model initialization completed in {end_time - start_time:.2f} seconds")
        
        return som_model, caption_model_processor
    except Exception as e:
        logger.error(f"Critical error during model initialization: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@log_execution_time
def process_image(image_data, box_threshold=0.03, request_id=None):
    logger.info(f"[Request ID: {request_id}] Starting image processing with box_threshold={box_threshold}")
    try:
        device = get_device()
        logger.info(f"[Request ID: {request_id}] Using device: {device}")
        
        # Convert base64 to PIL Image
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        logger.info(f"[Request ID: {request_id}] Image converted to RGB format: {image.size}")
        
        # Save temporary image
        temp_image_path = f"/tmp/temp_image_{request_id}.png"
        image.save(temp_image_path)
        
        # Memory management for CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            logger.info(f"[Request ID: {request_id}] Initial CUDA memory: {initial_memory/1024**2:.2f}MB")
        
        # Rest of the processing code remains the same
        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 3,
            'thickness': 3,
        }
        
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            temp_image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            temp_image_path,
            som_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.1
        )
        
        # Clean up
        try:
            os.remove(temp_image_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                logger.info(f"[Request ID: {request_id}] Final CUDA memory: {final_memory/1024**2:.2f}MB")
        except Exception as e:
            logger.warning(f"[Request ID: {request_id}] Cleanup error: {str(e)}")
        
        return {
            "labeled_image": labeled_img,
            "coordinates": label_coordinates,
            "parsed_content": parsed_content_list
        }
        
    except Exception as e:
        logger.error(f"[Request ID: {request_id}] Error processing image: {str(e)}")
        logger.error(f"[Request ID: {request_id}] Traceback: {traceback.format_exc()}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

# Initialize models at startup
logger.info("Starting serverless endpoint")
check_cuda_availability()
som_model, caption_model_processor = initialize_models()
logger.info("Models initialized successfully")

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
