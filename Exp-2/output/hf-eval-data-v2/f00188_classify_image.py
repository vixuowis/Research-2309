# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
import torch
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def classify_image(image_url):
    """
    Classify an image using the pretrained RegNetForImageClassification model.

    Args:
        image_url (str): The URL of the image to classify.

    Returns:
        str: The predicted label of the image.
    """
    # Load the pretrained model and feature extractor
    model = RegNetForImageClassification.from_pretrained('zuppif/regnet-y-040')
    feature_extractor = AutoFeatureExtractor.from_pretrained('zuppif/regnet-y-040')

    # Load the image from the URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Extract the features from the image
    inputs = feature_extractor(images=img, return_tensors='pt')

    # Classify the image
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label]

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.
    """
    # Define a test image URL
    test_image_url = 'https://example.com/test_image.jpg'

    # Call the classify_image function
    predicted_label = classify_image(test_image_url)

    # Assert that the function returns a string (the predicted label)
    assert isinstance(predicted_label, str)

# call_test_function_code --------------------

test_classify_image()