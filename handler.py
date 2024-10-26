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

# Configure logging with more detailed format
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler('/tmp/serverless_endpoint.log')  # File handler
    ]
)
logger = logging.getLogger(__name__)

# Add request ID tracking
def generate_request_id():
    return f"{int(time.time())}-{os.getpid()}-{hash(str(datetime.now()))}"

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def initialize_models():
    try:
        logger.info("Starting model initialization")
        start_time = time.time()
        
        device = 'cuda'
        logger.info(f"Using device: {device}")
        
        # Initialize SOM YOLO model
        logger.info("Loading YOLO model")
        som_model = get_yolo_model(model_path='icon_detect/best.pt')
        som_model.to(device)
        
        # Initialize caption model (Florence2)
        logger.info("Loading caption model and processor")
        caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="icon_caption_florence",
            device=device
        )
        
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
        # Log input image size
        raw_image_size = len(image_data)
        logger.info(f"[Request ID: {request_id}] Raw image data size: {raw_image_size} bytes")
        
        # Convert base64 to PIL Image
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        logger.info(f"[Request ID: {request_id}] Image converted to RGB format: {image.size}")
        
        # Save temporary image for OCR processing
        temp_image_path = f"/tmp/temp_image_{request_id}.png"
        image.save(temp_image_path)
        logger.info(f"[Request ID: {request_id}] Temporary image saved: {temp_image_path}")
        
        # Configure drawing settings
        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 3,
            'thickness': 3,
        }
        
        # Perform OCR
        logger.info(f"[Request ID: {request_id}] Starting OCR processing")
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            temp_image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        text, ocr_bbox = ocr_bbox_rslt
        logger.info(f"[Request ID: {request_id}] OCR completed. Found {len(text)} text elements")
        
        # Get labeled image and parsed content
        logger.info(f"[Request ID: {request_id}] Starting image labeling")
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
        logger.info(f"[Request ID: {request_id}] Image labeling completed. Found {len(label_coordinates)} objects")
        
        # Clean up temporary file
        try:
            os.remove(temp_image_path)
            logger.info(f"[Request ID: {request_id}] Temporary image removed")
        except Exception as e:
            logger.warning(f"[Request ID: {request_id}] Failed to remove temporary image: {str(e)}")
        
        return {
            "labeled_image": labeled_img,
            "coordinates": label_coordinates,
            "parsed_content": parsed_content_list
        }
        
    except Exception as e:
        logger.error(f"[Request ID: {request_id}] Error processing image: {str(e)}")
        logger.error(f"[Request ID: {request_id}] Traceback: {traceback.format_exc()}")
        raise

def handler(event):
    request_id = generate_request_id()
    logger.info(f"[Request ID: {request_id}] New request received")
    
    try:
        # Log input event (excluding image data for size considerations)
        event_log = event.copy()
        if "input" in event_log and "image" in event_log["input"]:
            event_log["input"]["image"] = f"<base64_image_{len(event_log['input']['image'])}bytes>"
        logger.info(f"[Request ID: {request_id}] Event received: {json.dumps(event_log)}")
        
        # Extract image data and parameters
        image_data = event.get("input", {}).get("image")
        box_threshold = event.get("input", {}).get("box_threshold", 0.03)
        
        if not image_data:
            logger.error(f"[Request ID: {request_id}] Missing required image data")
            raise ValueError("Image data is required")
        
        # Process the image
        logger.info(f"[Request ID: {request_id}] Processing image with box_threshold={box_threshold}")
        result = process_image(image_data, box_threshold, request_id)
        
        logger.info(f"[Request ID: {request_id}] Processing completed successfully")
        return {"output": result}
    
    except Exception as e:
        logger.error(f"[Request ID: {request_id}] Error in handler: {str(e)}")
        logger.error(f"[Request ID: {request_id}] Traceback: {traceback.format_exc()}")
        return {"error": str(e)}

# Initialize models at startup
logger.info("Initializing models at startup")
som_model, caption_model_processor = initialize_models()
logger.info("Models initialized successfully")

if __name__ == "__main__":
    logger.info("Starting serverless endpoint")
    runpod.serverless.start({"handler": handler})
