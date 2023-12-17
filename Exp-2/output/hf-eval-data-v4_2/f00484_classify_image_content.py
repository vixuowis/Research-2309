# requirements_file --------------------

!pip install -U pillow requests transformers torch

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch

# function_code --------------------

def classify_image_content(image_url):
    """
    Classify the content of the given image to determine if it contains a cat or a dog.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: 'cat' if the image contains a cat, 'dog' if the image contains a dog, otherwise 'unknown'.

    Raises:
        ValueError: If the image cannot be opened.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError(f'Image cannot be opened. {e}')

    inputs = processor(text=['a photo of a cat', 'a photo of a dog'], images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    if probs[0, 0] > probs[0, 1]:
        return 'cat'
    elif probs[0, 1] > probs[0, 0]:
        return 'dog'
    else:
        return 'unknown'

# test_function_code --------------------

def test_classify_image_content():
    print("Testing started.")
    cat_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'  # Known to be a cat image
    dog_url = 'https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg'  # Known to be a dog image
    unknown_url = 'https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg'  # Unknown content

    print("Testing case [1/3] started.")
    assert classify_image_content(cat_url) == 'cat', "Test case [1/3] failed: Incorrect classification for cat image."
    print("Testing case [2/3] started.")
    assert classify_image_content(dog_url) == 'dog', "Test case [2/3] failed: Incorrect classification for dog image."
    print("Testing case [3/3] started.")
    assert classify_image_content(unknown_url) == 'unknown', "Test case [3/3] failed: Incorrect classification for unknown image content."
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image_content()