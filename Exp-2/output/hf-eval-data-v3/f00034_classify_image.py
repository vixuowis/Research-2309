# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def classify_image(image_url: str, texts: list) -> dict:
    """
    Classify an image based on semantic similarity to the provided texts using a pre-trained Chinese CLIP model.

    Args:
        image_url (str): The URL of the image to be classified.
        texts (list): A list of text descriptions to compare the image against.

    Returns:
        dict: A dictionary containing the classification probabilities for each text description.
    """
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14-336px')
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=texts, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return probs

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function with a few test cases.
    """
    image_url = 'https://placekitten.com/200/300'
    texts = ['这是一只猫', '这是一只狗', '这是一只鸟']
    probs = classify_image(image_url, texts)
    assert isinstance(probs, dict)
    assert len(probs) == len(texts)
    assert all(isinstance(prob, float) for prob in probs.values())
    assert sum(probs.values()) - 1.0 < 1e-6
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()