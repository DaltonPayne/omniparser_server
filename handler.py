import runpod
import torch
from PIL import Image
from ultralytics import YOLO
import base64
import io
import logging
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_models():
    try:
        device = 'cuda'
        
        # Initialize SOM YOLO model
        som_model = get_yolo_model(model_path='./icon_detect/best.pt')
        som_model.to(device)
        
        # Initialize caption model (Florence2)
        caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="../icon_caption_florence",
            device=device
        )
        
        return som_model, caption_model_processor
    except Exception as e:
        logging.error(f"Error initializing models: {str(e)}")
        raise

# Initialize models at startup
som_model, caption_model_processor = initialize_models()

def process_image(image_data, box_threshold=0.03):
    try:
        # Convert base64 to PIL Image
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        
        # Save temporary image for OCR processing
        temp_image_path = "/tmp/temp_image.png"
        image.save(temp_image_path)
        
        # Configure drawing settings
        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 3,
            'thickness': 3,
        }
        
        # Perform OCR
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            temp_image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        # Get labeled image and parsed content
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
        
        return {
            "labeled_image": labeled_img,  # Base64 encoded image
            "coordinates": label_coordinates,
            "parsed_content": parsed_content_list
        }
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise

def handler(event):
    try:
        # Extract image data and parameters from the event
        image_data = event.get("input", {}).get("image")
        box_threshold = event.get("input", {}).get("box_threshold", 0.03)
        
        if not image_data:
            raise ValueError("Image data is required")
        
        # Process the image
        result = process_image(image_data, box_threshold)
        
        return {"output": result}
    
    except Exception as e:
        logging.error(f"Error in handler: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
