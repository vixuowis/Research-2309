from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
import requests


def classify_image(image_url):
    """
    Classify an image using the Swin Transformer model.

    Args:
        image_url (str): The URL of the image to classify.

    Returns:
        str: The predicted class of the image.
    """
    # Load the pre-trained Swin Transformer model for image classification
    model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
    feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')

    # Grab the image from the provided URL and load it into a PIL Image object
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Process the image using the feature extractor
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Pass the image features to the model for classification
    outputs = model(**inputs)
    logits = outputs.logits

    # Get the index of the class with the highest logit
    predicted_class_idx = logits.argmax(-1).item()

    # Return the predicted class
    return model.config.id2label[predicted_class_idx]