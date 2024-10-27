import runpod
import torch
from PIL import Image
from ultralytics import YOLO
import base64
import io
import logging
import os
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting Runpod serverless endpoint...")

def initialize_models():
    try:
        logging.info("Initializing models...")
        
        # Check for CUDA device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Device set to: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Set default tensor type to float32
        torch.set_default_tensor_type(torch.FloatTensor)
        
        # Initialize SOM YOLO model
        logging.info("Loading SOM YOLO model...")
        som_model = get_yolo_model(model_path='best.pt')
        som_model.to(device)
        som_model = som_model.float()
        logging.info("SOM YOLO model loaded and moved to device.")
        
        # Initialize caption model processor
        logging.info("Initializing caption model processor...")
        caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="icon_caption_florence",
            device=device
        )
        
        if hasattr(caption_model_processor, 'to'):
            caption_model_processor = caption_model_processor.to(dtype=torch.float32)
        
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
        logging.info("Image decoded and saved for processing.")
        
        # Configure drawing settings
        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 3,
            'thickness': 3,
        }
        
        # Perform OCR
        logging.info("Performing OCR...")
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            temp_image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        text, ocr_bbox = ocr_bbox_rslt
        logging.info("OCR processing completed.")
        logging.info(f"Detected text: {text}")
        
        # Get labeled image and parsed content
        logging.info("Getting labeled image and parsed content from SOM model...")
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
        logging.info("Labeled image and parsed content generated.")
        
        # Save the labeled image
        output_path = '/tmp/output_labeled.png'
        if isinstance(labeled_img, str):  # If it's base64
            img_data = base64.b64decode(labeled_img)
            with open(output_path, 'wb') as f:
                f.write(img_data)
        else:  # If it's a PIL Image
            labeled_img.save(output_path)
        
        # Read the output image back as base64
        with open(output_path, 'rb') as f:
            output_image_base64 = base64.b64encode(f.read()).decode('ascii')
        
        logging.info(f"Labeled image processed")
        logging.info(f"Detected coordinates: {label_coordinates}")
        logging.info(f"Parsed content: {parsed_content_list}")
        
        return {
            "labeled_image": output_image_base64,
            "labeled_image_path": output_path,
            "coordinates": label_coordinates,
            "parsed_content": parsed_content_list,
            "detected_text": text,
            "ocr_boxes": ocr_bbox
        }
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if os.path.exists(output_path):
            os.remove(output_path)

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
        
        # Process the image using the global models
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
