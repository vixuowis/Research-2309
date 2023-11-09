from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests


def detect_objects(image_url):
    """
    This function detects objects in an image using the YOLOS Tiny model from Hugging Face Transformers.
    
    Parameters:
    image_url (str): The URL of the image to detect objects in.
    
    Returns:
    dict: A dictionary containing the logits and predicted bounding boxes for the detected objects.
    """
    # Load the pretrained YOLOS Tiny model and feature extractor
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    
    # Load the image from the specified URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Prepare the input tensors for the model
    inputs = feature_extractor(images=image, return_tensors='pt')
    
    # Pass the input tensors to the model to get the object detection outputs
    outputs = model(**inputs)
    
    # Extract the logits and predicted bounding boxes from the outputs
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    
    return {'logits': logits, 'bboxes': bboxes}