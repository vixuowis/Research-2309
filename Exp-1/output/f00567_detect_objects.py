from typing import *
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

def detect_objects(image):
    image_processor = AutoImageProcessor.from_pretrained("devonho/detr-resnet-50_finetuned_cppe5")
    model = AutoModelForObjectDetection.from_pretrained("devonho/detr-resnet-50_finetuned_cppe5")

    # Detect objects in the image
    # Parameters:
    #     image (PIL.Image.Image): The input image
    # Returns:
    #     results (list): List of detected objects with their confidence and location
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    return results
