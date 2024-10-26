# from ultralytics import YOLO
import os
import io
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import json
import requests
# utility function
import os
from openai import AzureOpenAI

import json
import sys
import os
import cv2
import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt
import easyocr
reader = easyocr.Reader(['en'])
import time
import base64

import os
import ast
import torch
from typing import Tuple, List
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T


def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map=None, 
            torch_dtype=torch.float32
        ).to(device)
    elif model_name == "florence2":
        from transformers import AutoProcessor, AutoModelForCausalLM 
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        
        # Force CPU initialization in float32, then move to device
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,  # Always use float32
            trust_remote_code=True,
            device_map=None  # Prevent automatic device placement
        )
        
        # Move to device after initialization
        model = model.to(device)
        
        # If using CUDA and want to use half precision after initialization
        if device == 'cuda':
            model = model.half()  # Convert to half precision after moving to GPU
    
    return {'model': model, 'processor': processor}


def get_yolo_model(model_path):
    from ultralytics import YOLO
    # Load the model.
    model = YOLO(model_path)
    return model


def get_parsed_content_icon(filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=None):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"

    batch_size = 10  # Number of samples per batch
    generated_texts = []
    device = model.device

    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i:i+batch_size]
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
        if 'florence' in model.config.name_or_path:
            generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=1024,num_beams=3, do_sample=False)
        else:
            generated_ids = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1) # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)

    return generated_texts



def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}] 
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i+batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs ={'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
            inputs['input_ids'].append(input['input_ids'])
            inputs['attention_mask'].append(input['attention_mask'])
            inputs['pixel_values'].append(input['pixel_values'])
            inputs['image_sizes'].append(input['image_sizes'])
        max_len = max([x.shape[1] for x in inputs['input_ids']])
        for i, v in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = torch.cat([processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
            inputs['attention_mask'][i] = torch.cat([torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1)
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = { 
            "max_new_tokens": 25, 
            "temperature": 0.01, 
            "do_sample": False, 
        } 
        generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
        # # remove input tokens 
        generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [res.strip('\n').strip() for res in response]
        generated_texts.extend(response)

    return generated_texts

def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                if not any(IoU(box1, box3) > iou_threshold for k, box3 in enumerate(ocr_bbox)):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed.
    """
    h, w, _ = image_source.shape
    
    # Convert boxes to numpy and ensure they're valid
    if boxes is None or len(boxes) == 0:
        return image_source.copy(), {}
        
    # Scale boxes if they're in normalized coordinates
    boxes = boxes * torch.Tensor([w, h, w, h])
    
    # Convert to xyxy format for detection
    try:
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.numpy()
    except Exception as e:
        print(f"Error converting boxes: {e}")
        return image_source.copy(), {}

    # Convert to xywh format for return values
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh")
    if isinstance(xywh, torch.Tensor):
        xywh = xywh.numpy()

    # Create detections object
    detections = []
    for box in xyxy:
        if all(x is not None for x in box):  # Check for None values
            detections.append(box.tolist() if isinstance(box, np.ndarray) else box)

    # Create labels list
    labels = [str(phrase) for phrase in phrases[:len(detections)]]

    # Initialize box annotator
    box_annotator = BoxAnnotator(
        text_scale=text_scale,
        text_padding=text_padding,
        text_thickness=text_thickness,
        thickness=thickness
    )

    # Annotate the image
    annotated_frame = image_source.copy()
    if detections:
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

    # Create label coordinates dictionary
    label_coordinates = {
        str(phrase): coords 
        for phrase, coords in zip(phrases, xywh) 
        if all(x is not None for x in coords)
    }

    return annotated_frame, label_coordinates


def predict(model, image, caption, box_threshold, text_threshold):
    """ Use huggingface model to replace the original model
    """
    model, processor = model['model'], model['processor']
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold, # 0.4,
        text_threshold=text_threshold, # 0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


def predict_yolo(model, image_path, box_threshold):
    """ Use huggingface model to replace the original model
    """
    # model = model['model']
    
    result = model.predict(
    source=image_path,
    conf=box_threshold,
    # iou=0.5, # default 0.7
    )
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases


def get_som_labeled_img(img_path, model=None, BOX_TRESHOLD = 0.01, output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None, caption_model_processor=None, ocr_text=[], use_local_semantics=True, iou_threshold=0.9,prompt=None):
    """ ocr_bbox: list of xyxy format bbox
    """
    TEXT_PROMPT = "clickable buttons on the screen"
    # BOX_TRESHOLD = 0.02 # 0.05/0.02 for web and 0.1 for mobile
    TEXT_TRESHOLD = 0.01 # 0.9 # 0.01
    image_source = Image.open(img_path).convert("RGB")
    w, h = image_source.size
    # import pdb; pdb.set_trace()
    if False: # TODO
        xyxy, logits, phrases = predict(model=model, image=image_source, caption=TEXT_PROMPT, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD)
    else:
        xyxy, logits, phrases = predict_yolo(model=model, image_path=img_path, box_threshold=BOX_TRESHOLD)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    # annotate the image with labels
    h, w, _ = image_source.shape
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox=ocr_bbox.tolist()
    else:
        print('no ocr bbox!!!')
        ocr_bbox = None
    filtered_boxes = remove_overlap(boxes=xyxy, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox)
    
    # get parsed icon local semantics
    if use_local_semantics:
        caption_model = caption_model_processor['model']
        if 'phi3_v' in caption_model.config.model_type: 
            parsed_content_icon = get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor)
        else:
            parsed_content_icon = get_parsed_content_icon(filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=prompt)
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)
        parsed_content_icon_ls = []
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
        parsed_content_merged = ocr_text + parsed_content_icon_ls
    else:
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    phrases = [i for i in range(len(filtered_boxes))]
    
    # draw boxes
    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
    else:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)
    
    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    if output_coord_in_ratio:
        # h, w, _ = image_source.shape
        label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return encoded_image, label_coordinates, parsed_content_merged


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h
    


def check_ocr_box(image_path, display_img = True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None):
    if easyocr_args is None:
        easyocr_args = {}
    result = reader.readtext(image_path, **easyocr_args)
    is_goal_filtered = False
    # print('goal filtering pred:', result[-5:])
    coord = [item[0] for item in result]
    text = [item[1] for item in result]
    # read the image using cv2
    if display_img:
        opencv_img = cv2.imread(image_path)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            # print(x, y, a, b)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
        
        # Display the image
        plt.imshow(opencv_img)
    else:
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]
        # print('bounding box!!!', bb)
    return (text, bb), is_goal_filtered


import cv2
import numpy as np
from typing import List, Tuple, Union, Optional

class Colors:
    def __init__(self):
        hex_colors = [
            "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
            "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#2C99A8", "#00C2FF",
            "#344593", "#6473FF", "#0018EC", "#8438FF", "#520085", "#CB38FF"
        ]
        self.palette = [self._hex2rgb(c) for c in hex_colors]
        self.n = len(self.palette)

    def _hex2rgb(self, h: str) -> Tuple[int, int, int]:
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    def __call__(self, i: int) -> Tuple[int, int, int]:
        return self.palette[i % self.n]

class BoxAnnotator:
    def __init__(
        self,
        color: Union[Tuple[int, int, int], Colors] = Colors(),
        thickness: int = 2,
        text_thickness: int = 2,
        text_scale: float = 0.5,
        text_padding: int = 10,
        text_color: Tuple[int, int, int] = (255, 255, 255)
    ):
        self.color = color
        self.thickness = thickness
        self.text_thickness = text_thickness
        self.text_scale = text_scale
        self.text_padding = text_padding
        self.text_color = text_color

    def _validate_detection(self, detection: Tuple) -> bool:
        """
        Validates that a detection tuple contains valid coordinates.
        """
        if detection is None or len(detection) < 4:
            return False
        return all(x is not None and isinstance(x, (int, float)) for x in detection[:4])

    def annotate(
        self,
        scene: np.ndarray,
        detections: List[Optional[Tuple]],
        labels: Optional[List[str]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        skip_label: bool = False
    ) -> np.ndarray:
        """
        Annotates the scene with bounding boxes and labels.
        
        Args:
            scene: The image to annotate
            detections: List of detections, each in format (x1, y1, x2, y2, score)
            labels: List of labels corresponding to each detection
            image_size: Optional tuple of (width, height) to scale coordinates
            skip_label: If True, don't draw labels
        """
        if scene is None:
            raise ValueError("Scene cannot be None")
            
        frame = scene.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Scale coordinates if image_size is provided
        if image_size:
            scale_w, scale_h = image_size[0] / frame.shape[1], image_size[1] / frame.shape[0]
        else:
            scale_w, scale_h = 1.0, 1.0
        
        valid_detections = []
        valid_labels = []
        
        # First, validate all detections and labels
        for i, detection in enumerate(detections or []):
            if self._validate_detection(detection):
                valid_detections.append(detection)
                if labels and i < len(labels):
                    valid_labels.append(labels[i])
                elif labels:
                    valid_labels.append("")
        
        # Now process only valid detections
        for i, detection in enumerate(valid_detections):
            try:
                x1, y1, x2, y2 = map(float, detection[:4])
                
                # Scale coordinates
                x1, y1, x2, y2 = map(int, [
                    x1 * scale_w,
                    y1 * scale_h,
                    x2 * scale_w,
                    y2 * scale_h
                ])
                
                # Get color for this detection
                color = self.color(i) if isinstance(self.color, Colors) else self.color
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
                
                # Draw label if provided and not skipped
                if valid_labels and i < len(valid_labels) and not skip_label:
                    text = str(valid_labels[i])
                    
                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, font, self.text_scale, self.text_thickness
                    )
                    
                    # Calculate text background rectangle
                    padding = self.text_padding
                    text_x1 = x1
                    text_y1 = y1 - text_height - padding
                    text_x2 = x1 + text_width + padding
                    text_y2 = y1
                    
                    # Ensure the text background stays within the image
                    text_y1 = max(text_y1, 0)
                    if text_y1 == 0:
                        text_y2 = text_height + padding
                    
                    # Draw text background
                    cv2.rectangle(frame, (text_x1, text_y1), (text_x2, text_y2), color, -1)
                    
                    # Draw text
                    cv2.putText(
                        frame, text,
                        (x1 + padding // 2, y1 - padding // 2),
                        font, self.text_scale, self.text_color, self.text_thickness
                    )
            except (TypeError, ValueError, IndexError) as e:
                print(f"Warning: Failed to process detection {i}: {e}")
                continue
        
        return frame
