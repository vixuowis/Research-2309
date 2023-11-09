# function_import --------------------

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# function_code --------------------

def classify_image(image_path: str, categories: list) -> dict:
    """
    Classify an image into given categories using a pre-trained CLIP model.

    Args:
        image_path (str): The path to the image that needs to be classified.
        categories (list): The list of categories we want to classify the image into.

    Returns:
        dict: A dictionary where keys are categories and values are corresponding probabilities.
    """
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

    image = Image.open(image_path)
    inputs = processor(text=categories, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    probs = probs.tolist()[0]

    return {categories[i]: probs[i] for i in range(len(categories))}

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.
    """
    image_path = 'test_image.jpg'
    categories = ['a photo of a cat', 'a photo of a dog', 'a photo of a bird']
    result = classify_image(image_path, categories)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert len(result) == len(categories), 'The result should contain probabilities for all categories.'
    assert all(isinstance(value, float) for value in result.values()), 'All probabilities should be floats.'
    assert abs(sum(result.values()) - 1) < 1e-6, 'The probabilities should sum up to 1.'

# call_test_function_code --------------------

test_classify_image()