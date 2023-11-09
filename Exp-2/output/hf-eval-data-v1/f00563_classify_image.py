from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests


def classify_image(image_url):
    """
    This function classifies the image from the provided URL using the pre-trained model 'google/mobilenet_v1_0.75_192'.
    
    Parameters:
    image_url (str): The URL of the image to be classified.
    
    Returns:
    str: The predicted class of the image.
    """
    # Load the image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    
    # Load the pre-processor and the model
    preprocessor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')
    model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192')
    
    # Preprocess the image and feed it to the model
    inputs = preprocessor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    
    # Get the predicted class index
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Return the name of the predicted class
    return model.config.id2label[predicted_class_idx]