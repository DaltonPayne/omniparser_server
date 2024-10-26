import runpod
import torch
from PIL import Image
from ultralytics import YOLO
import base64
import io
import logging
import time
import traceback
import json
from datetime import datetime
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add file handler for persistent logging
file_handler = logging.FileHandler('/tmp/serverless_endpoint.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
))
logger.addHandler(file_handler)

class RequestContext:
    """Context manager for tracking request metrics"""
    def __init__(self, request_id):
        self.request_id = request_id
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Request {self.request_id} started at {datetime.now().isoformat()}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logger.info(f"Request {self.request_id} completed in {duration:.2f} seconds")
        if exc_type:
            logger.error(f"Request {self.request_id} failed with error: {exc_val}")
            logger.error(f"Traceback: {''.join(traceback.format_tb(exc_tb))}")

def log_memory_usage():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

def initialize_models():
    """Initialize ML models with comprehensive logging"""
    logger.info("Starting model initialization")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        start_time = time.time()
        
        # Initialize SOM YOLO model
        logger.info("Loading SOM YOLO model")
        som_model = get_yolo_model(model_path='icon_detect/best.pt')
        som_model.to(device)
        logger.info("SOM YOLO model loaded successfully")
        
        # Initialize caption model
        logger.info("Loading caption model (Florence2)")
        caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="icon_caption_florence",
            device=device
        )
        logger.info("Caption model loaded successfully")
        
        initialization_time = time.time() - start_time
        logger.info(f"Model initialization completed in {initialization_time:.2f} seconds")
        log_memory_usage()
        
        return som_model, caption_model_processor
    except Exception as e:
        logger.error(f"Critical error during model initialization: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Initialize models at startup
logger.info("Starting serverless endpoint initialization")
som_model, caption_model_processor = initialize_models()
logger.info("Serverless endpoint initialization completed")

def process_image(image_data, box_threshold=0.03, request_id=None):
    """Process image with detailed logging of each step"""
    logger.info(f"Starting image processing for request {request_id}")
    try:
        # Log input parameters
        logger.info(f"Processing image with box_threshold: {box_threshold}")
        
        # Track step timing
        step_times = {}
        
        # Convert base64 to PIL Image
        start_time = time.time()
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
            logger.info(f"Image decoded successfully - Size: {image.size}")
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {str(e)}")
            raise
        step_times['image_decode'] = time.time() - start_time
        
        # Save temporary image
        start_time = time.time()
        temp_image_path = f"/tmp/temp_image_{request_id}.png"
        image.save(temp_image_path)
        logger.info(f"Temporary image saved at: {temp_image_path}")
        step_times['save_temp'] = time.time() - start_time
        
        # Configure drawing settings
        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 3,
            'thickness': 3,
        }
        
        # Perform OCR
        start_time = time.time()
        logger.info("Starting OCR processing")
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            temp_image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        text, ocr_bbox = ocr_bbox_rslt
        logger.info(f"OCR completed - Found {len(text)} text regions")
        step_times['ocr'] = time.time() - start_time
        
        # Get labeled image
        start_time = time.time()
        logger.info("Starting image labeling and parsing")
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
        logger.info(f"Image labeling completed - Found {len(label_coordinates)} objects")
        step_times['labeling'] = time.time() - start_time
        
        # Log performance metrics
        logger.info(f"Step timing breakdown: {json.dumps(step_times, indent=2)}")
        log_memory_usage()
        
        return {
            "labeled_image": labeled_img,
            "coordinates": label_coordinates,
            "parsed_content": parsed_content_list
        }
        
    except Exception as e:
        logger.error(f"Error processing image for request {request_id}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def handler(event):
    """Main handler with request tracking and comprehensive logging"""
    request_id = f"req_{int(time.time()*1000)}"
    
    with RequestContext(request_id):
        try:
            logger.info(f"Received event: {json.dumps(event.get('input', {}), indent=2)}")
            
            # Extract and validate input
            image_data = event.get("input", {}).get("image")
            box_threshold = event.get("input", {}).get("box_threshold", 0.03)
            
            if not image_data:
                logger.error("Missing required input: image data")
                raise ValueError("Image data is required")
            
            # Process image
            result = process_image(image_data, box_threshold, request_id)
            
            # Log success
            logger.info(f"Successfully processed request {request_id}")
            return {"output": result}
        
        except Exception as e:
            logger.error(f"Handler error for request {request_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting serverless endpoint")
    runpod.serverless.start({"handler": handler})
