from transformers import YolosForObjectDetection, YolosFeatureExtractor
from PIL import Image
import requests


def detect_objects(image_path):
    """
    This function uses the YOLOS model fine-tuned on COCO 2017 object detection to detect objects in an image.
    The model is loaded from Hugging Face Transformers.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    dict: The detected objects and their bounding boxes.
    """
    # Load the image data
    image = Image.open(requests.get(image_path, stream=True).raw)
    
    # Load the pre-trained model
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
    
    # Load the feature extractor
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
    
    # Prepare the inputs
    inputs = feature_extractor(images=image, return_tensors='pt')
    
    # Get the model outputs
    outputs = model(**inputs)
    
    # Return the logits and predicted boxes
    return {'logits': outputs.logits, 'bboxes': outputs.pred_boxes}