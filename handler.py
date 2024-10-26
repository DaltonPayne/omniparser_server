import runpod
import torch
from PIL import Image
from ultralytics import YOLO
import base64
import io
import logging
import time
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
        logger.debug(traceback.format_exc())
        return False

def get_device():
    """Get the appropriate device for model operations"""
    try:
        if check_cuda_availability():
            return torch.device('cuda')
        return torch.device('cpu')
    except Exception as e:
        logger.error("Error in get_device function")
        logger.debug(traceback.format_exc())
        raise

def initialize_models():
    """Initialize models and handle potential errors"""
    try:
        logger.info("Starting model initialization")
        start_time = time.time()
        
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load YOLO model
        try:
            logger.info("Loading YOLO model")
            som_model = get_yolo_model(model_path='icon_detect/best.pt')
            som_model.to(device)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error("Error loading YOLO model")
            logger.debug(traceback.format_exc())
            raise RuntimeError("Failed to load YOLO model") from e
        
        # Load caption model
        try:
            logger.info("Loading caption model and processor")
            caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path="icon_caption_florence",
                device=device
            )
            logger.info("Caption model and processor loaded successfully")
        except Exception as e:
            logger.error("Error loading caption model or processor")
            logger.debug(traceback.format_exc())
            raise RuntimeError("Failed to load caption model or processor") from e
        
        # Verify model devices
        try:
            logger.info(f"YOLO model device: {next(som_model.parameters()).device}")
            logger.info(f"Caption model device: {next(caption_model_processor['model'].parameters()).device}")
        except Exception as e:
            logger.error("Error verifying model devices")
            logger.debug(traceback.format_exc())
            raise RuntimeError("Device verification failed") from e

        end_time = time.time()
        logger.info(f"Model initialization completed in {end_time - start_time:.2f} seconds")
        
        return som_model, caption_model_processor
    except Exception as e:
        logger.critical("Critical error during model initialization")
        logger.debug(traceback.format_exc())
        raise

def log_execution_time(func):
    """Decorator to log execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in function {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
        finally:
            end_time = time.time()
            logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
    return wrapper

@log_execution_time
def process_image(image_data, box_threshold=0.03, request_id=None):
    logger.info(f"[Request ID: {request_id}] Starting image processing with box_threshold={box_threshold}")
    try:
        device = get_device()
        logger.info(f"[Request ID: {request_id}] Using device: {device}")
        
        # Convert base64 to PIL Image
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
            logger.info(f"[Request ID: {request_id}] Image converted to RGB format: {image.size}")
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Failed to convert base64 to image")
            logger.debug(traceback.format_exc())
            raise ValueError("Invalid image data") from e
        
        # Save temporary image
        temp_image_path = f"/tmp/temp_image_{request_id}.png"
        try:
            image.save(temp_image_path)
            logger.info(f"[Request ID: {request_id}] Temporary image saved at {temp_image_path}")
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Failed to save temporary image")
            logger.debug(traceback.format_exc())
            raise IOError("Failed to save temporary image") from e
        
        # Memory management for CUDA
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                logger.info(f"[Request ID: {request_id}] Initial CUDA memory: {initial_memory / 1024 ** 2:.2f}MB")
            except Exception as e:
                logger.warning(f"[Request ID: {request_id}] CUDA memory management error: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Process with models
        try:
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
            logger.info(f"[Request ID: {request_id}] Image processed successfully")
        
        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Error during image processing")
            logger.debug(traceback.format_exc())
            raise
        
        # Clean up
        try:
            os.remove(temp_image_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                logger.info(f"[Request ID: {request_id}] Final CUDA memory: {final_memory / 1024 ** 2:.2f}MB")
            logger.info(f"[Request ID: {request_id}] Temporary image and memory cleanup complete")
        except Exception as e:
            logger.warning(f"[Request ID: {request_id}] Cleanup error: {str(e)}")
            logger.debug(traceback.format_exc())
        
        return {
            "labeled_image": labeled_img,
            "coordinates": label_coordinates,
            "parsed_content": parsed_content_list
        }
        
    except Exception as e:
        logger.critical(f"[Request ID: {request_id}] Critical error in process_image function")
        logger.debug(traceback.format_exc())
        raise

# Initialize models at startup
logger.info("Starting serverless endpoint")
try:
    check_cuda_availability()
    som_model, caption_model_processor = initialize_models()
    logger.info("Models initialized successfully")
except Exception as e:
    logger.critical("Failed to initialize models, exiting")
    logger.debug(traceback.format_exc())
    exit(1)

if __name__ == "__main__":
    runpod.serverless.start({"handler": process_image})
