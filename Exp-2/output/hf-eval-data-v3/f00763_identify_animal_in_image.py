# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def identify_animal_in_image(image_url):
    """
    Identify whether the image contains a cat or a dog based on the Chinese language image captions.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The identified animal in the image, either '猫' (cat) or '狗' (dog).
    """
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    image = Image.open(requests.get(image_url, stream=True).raw)
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
    Test the function identify_animal_in_image.
    """
    cat_image_url = 'https://placekitten.com/200/300'
    dog_image_url = 'https://placedog.net/500'
    assert identify_animal_in_image(cat_image_url) == '猫'
    assert identify_animal_in_image(dog_image_url) == '狗'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_identify_animal_in_image()