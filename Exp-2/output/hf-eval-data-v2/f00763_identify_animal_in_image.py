# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

# function_code --------------------

def identify_animal_in_image(url):
    """
    Identify whether the image at the given URL contains a cat or a dog using a pre-trained ChineseCLIPModel.

    Args:
        url (str): The URL of the image to classify.

    Returns:
        str: The identified animal ('猫' for cat, '狗' for dog).
    """
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    image = Image.open(requests.get(url, stream=True).raw)
    texts = ['猫', '狗']
    inputs = processor(images=image, text=texts, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    highest_prob_idx = probs.argmax(dim=1)
    return texts[highest_prob_idx]

# test_function_code --------------------

def test_identify_animal_in_image():
    """
    Test the identify_animal_in_image function with a cat and a dog image.
    """
    cat_url = 'https://example.com/cat.jpg'
    dog_url = 'https://example.com/dog.jpg'
    assert identify_animal_in_image(cat_url) == '猫'
    assert identify_animal_in_image(dog_url) == '狗'

# call_test_function_code --------------------

test_identify_animal_in_image()