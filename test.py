import torch
from PIL import Image
import logging
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import base64
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting model test script...")

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

def process_single_image(image_path, som_model, caption_model_processor, box_threshold=0.03):
    try:
        logging.info(f"Processing image: {image_path}")
        
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        logging.info("Image loaded and converted to RGB.")
        
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
            image_path,
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
            image_path,
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
        output_path = 'output_labeled.png'
        if isinstance(labeled_img, str):  # If it's base64
            img_data = base64.b64decode(labeled_img)
            with open(output_path, 'wb') as f:
                f.write(img_data)
        else:  # If it's a PIL Image
            labeled_img.save(output_path)
        
        logging.info(f"Labeled image saved to: {output_path}")
        logging.info(f"Detected coordinates: {label_coordinates}")
        logging.info(f"Parsed content: {parsed_content_list}")
        
        return {
            "labeled_image_path": output_path,
            "coordinates": label_coordinates,
            "parsed_content": parsed_content_list
        }
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise

def main():
    try:
        # Initialize models
        som_model, caption_model_processor = initialize_models()
        logging.info("Models initialized successfully.")
        
        # Test image path - replace with your test image path
        test_image_path = "screenshot.png"  # Replace with your image path
        
        # Process the image
        result = process_single_image(test_image_path, som_model, caption_model_processor)
        
        # Print results
        print("\nProcessing Results:")
        print(f"Output image saved to: {result['labeled_image_path']}")
        print("\nDetected Coordinates:")
        for coord in result['coordinates']:
            print(coord)
        print("\nParsed Content:")
        for content in result['parsed_content']:
            print(content)
            
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
