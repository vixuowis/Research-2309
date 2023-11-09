# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def image_geolocalization(url, choices):
    """
    This function uses a pretrained CLIP model to perform image geolocalization.
    It takes an image URL and a list of location choices as input, and returns the estimated probabilities for each location.

    Args:
        url (str): The URL of the image to be geolocalized.
        choices (list): A list of possible locations for the image.

    Returns:
        probs (torch.Tensor): A tensor containing the estimated probabilities for each location.
    """
    model = CLIPModel.from_pretrained('geolocal/StreetCLIP')
    processor = CLIPProcessor.from_pretrained('geolocal/StreetCLIP')
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=choices, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs

# test_function_code --------------------

def test_image_geolocalization():
    """
    This function tests the image_geolocalization function with a sample image and location choices.
    It asserts that the output is a tensor and that the sum of the probabilities is approximately 1.
    """
    url = 'https://image_url_here.jpeg'
    choices = ['San Jose', 'San Diego', 'Los Angeles', 'Las Vegas', 'San Francisco']
    probs = image_geolocalization(url, choices)
    assert isinstance(probs, torch.Tensor)
    assert abs(probs.sum().item() - 1) < 1e-6

# call_test_function_code --------------------

test_image_geolocalization()