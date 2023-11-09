from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests

# Function to detect objects in an image
# @param: url - URL of the image
# @return: logits, bboxes - Predicted objects and their bounding boxes

def detect_objects(url):
    # Open the image
    image = Image.open(requests.get(url, stream=True).raw)

    # Load the pre-trained model
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-dc5')
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')

    # Extract features from the image
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Get predictions from the model
    outputs = model(**inputs)

    # Get the predicted objects and their bounding boxes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    return logits, bboxes