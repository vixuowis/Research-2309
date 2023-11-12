# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image(image_url):
    """
    Classify an image as a cat or a dog using the pretrained CLIP model.

    Args:
        image_url (str): The URL of the image to classify.

    Returns:
        str: The classification result ('cat' or 'dog').

    Raises:
        requests.exceptions.RequestException: If there is an error to download the image.
        urllib3.exceptions.ProtocolError: If there is a connection error.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=['a photo of a cat', 'a photo of a dog'], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return 'cat' if probs[0] > probs[1] else 'dog'

# test_function_code --------------------

def test_classify_image():
    """Test the classify_image function."""
    cat_url = 'https://placekitten.com/200/300'
    dog_url = 'https://placedog.net/500'
    assert classify_image(cat_url) == 'cat'
    assert classify_image(dog_url) == 'dog'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()