from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

def predict_car_brand(url):
    """
    This function predicts the car brand from an image.

    Args:
        url (str): The URL or file path of the car's image.

    Returns:
        str: The predicted car brand.
    """
    image = Image.open(requests.get(url, stream=True).raw)
    processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
    model = AutoModelForImageClassification.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]