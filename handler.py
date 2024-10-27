import runpod
import torch
from PIL import Image
from ultralytics import YOLO
import base64
import io
import logging
import os
from utils import check_ocr_box, get_caption_model_processor, get_yolo_model
from torchvision.ops import box_convert

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting Runpod serverless endpoint...")

def initialize_models():
    try:
        logging.info("Initializing models...")
        
        # Check for CUDA device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Device set to: {device}")
        
        # Initialize YOLO model
        logging.info("Loading YOLO model...")
        som_model = get_yolo_model(model_path='best.pt')
        som_model.to(device)
        som_model = som_model.float()
        logging.info("YOLO model loaded and moved to device.")
        
        # Initialize caption model processor with explicit float32
        logging.info("Initializing caption model processor...")
        with torch.cuda.amp.autocast(enabled=False):  # Disable automatic mixed precision
            caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path="icon_caption_florence",
                device=device
            )
        logging.info("Caption model processor initialized successfully.")
        
        return som_model, caption_model_processor
    except Exception as e:
        logging.error(f"Error initializing models: {str(e)}")
        raise

def process_image(image_data, som_model, caption_model_processor, box_threshold=0.03):
    try:
        logging.info("Processing image data...")
        
        # Create temp directory if it doesn't exist
        os.makedirs("/tmp", exist_ok=True)
        
        # Convert base64 to PIL Image and save temporarily
        temp_image_path = "/tmp/temp_image.png"
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        image.save(temp_image_path)
        
        # Get image dimensions
        w, h = image.size
        
        # Perform OCR
        logging.info("Performing OCR...")
        ocr_result, _ = check_ocr_box(
            temp_image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        ocr_text, ocr_boxes = ocr_result
        
        # Perform YOLO detection
        logging.info("Performing YOLO detection...")
        results = som_model.predict(
            source=temp_image_path,
            conf=box_threshold,
        )
        
        # Extract boxes and convert to proper format
        yolo_boxes = results[0].boxes.xyxy.cpu()
        
        # Convert coordinates to normalized format
        normalized_yolo_boxes = yolo_boxes / torch.tensor([w, h, w, h])
        
        # Process OCR boxes to match format
        normalized_ocr_boxes = []
        for box in ocr_boxes:
            norm_box = [
                box[0] / w,  # x1
                box[1] / h,  # y1
                box[2] / w,  # x2
                box[3] / h   # y2
            ]
            normalized_ocr_boxes.append(norm_box)
        
        # Get icon captions
        icon_boxes = normalized_yolo_boxes.tolist()
        icon_descriptions = []
        
        for i, box in enumerate(icon_boxes):
            # Extract region
            x1, y1, x2, y2 = [int(coord * (w if i % 2 == 0 else h)) for i, coord in enumerate(box)]
            icon_image = image.crop((x1, y1, x2, y2))
            
            # Get caption
            inputs = caption_model_processor['processor'](
                images=icon_image, 
                text="<CAPTION>", 
                return_tensors="pt"
            ).to(caption_model_processor['model'].device)
            
            outputs = caption_model_processor['model'].generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128
            )
            
            caption = caption_model_processor['processor'].batch_decode(
                outputs, 
                skip_special_tokens=True
            )[0].strip()
            
            icon_descriptions.append(caption)
        
        # Combine results
        elements = []
        
        # Add OCR text elements
        for i, (box, text) in enumerate(zip(normalized_ocr_boxes, ocr_text)):
            elements.append({
                "type": "text",
                "id": f"text_{i}",
                "coordinates": box,
                "content": text
            })
        
        # Add icon elements
        for i, (box, desc) in enumerate(zip(icon_boxes, icon_descriptions)):
            elements.append({
                "type": "icon",
                "id": f"icon_{i}",
                "coordinates": box,
                "content": desc
            })
        
        return {
            "elements": elements,
            "image_size": {
                "width": w,
                "height": h
            }
        }
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# Initialize models globally
try:
    som_model, caption_model_processor = initialize_models()
    logging.info("Model initialization complete.")
except Exception as init_error:
    print(f"Initialization Error: {init_error}")
    logging.error(f"Critical error during model initialization: {init_error}")
    raise

def handler(event):
    try:
        logging.info("Handler invoked with event data.")
        
        # Extract image data and parameters from the event
        image_data = event.get("input", {}).get("image")
        box_threshold = event.get("input", {}).get("box_threshold", 0.03)
        
        if not image_data:
            logging.error("No image data found in request.")
            raise ValueError("Image data is required")
        
        # Process the image
        result = process_image(image_data, som_model, caption_model_processor, box_threshold)
        logging.info("Image processing completed successfully.")
        
        return {
            "output": result
        }
    
    except Exception as e:
        logging.error(f"Error in handler: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        logging.info("Starting Runpod serverless endpoint.")
        runpod.serverless.start({"handler": handler})
    except Exception as main_error:
        print(f"Startup Error: {main_error}")
        logging.error(f"Error during Runpod serverless start: {main_error}")
        raise
