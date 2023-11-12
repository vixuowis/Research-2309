# function_import --------------------

import requests
from PIL import Image
from io import BytesIO
from transformers import ViTFeatureExtractor, ViTForImageClassification

# function_code --------------------

def is_adult(image_url):
    """
    Determine if a person in an image is an adult using a pretrained model.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        bool: True if the person in the image is an adult, False otherwise.

    Raises:
        PIL.UnidentifiedImageError: If the image cannot be identified and opened.
    """
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
    inputs = transforms(image, return_tensors='pt')
    output = model(**inputs)
    proba = output.logits.softmax(1)
    predicted_age_class = proba.argmax(1)
    return predicted_age_class >= 18

# test_function_code --------------------

def test_is_adult():
    """
    Test the is_adult function with various test cases.
    """
    assert is_adult('https://placekitten.com/200/300') == False
    assert is_adult('https://some-adult-image-url-here.jpg') == True
    assert is_adult('https://some-child-image-url-here.jpg') == False
    return 'All Tests Passed'

# call_test_function_code --------------------

test_is_adult()