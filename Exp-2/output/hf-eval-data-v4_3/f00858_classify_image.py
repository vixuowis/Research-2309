# requirements_file --------------------

import subprocess

requirements = ["Pillow", "requests", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# function_code --------------------

def classify_image(image_url: str, labels: List[str]) -> Dict[str, float]:
    """
    Classify an image using zero-shot learning with a pre-trained CLIP model.

    Args:
        image_url: A string URL of the image to be classified.
        labels: A list of strings representing the categories for classification.
    
    Returns:
        A dictionary where keys are the labels and values are the probabilities.
    
    Raises:
        ValueError: If the image URL is not valid or cannot be opened.
    """
    model = CLIPModel.from_pretrained('flax-community/clip-rsicd-v2')
    processor = CLIPProcessor.from_pretrained('flax-community/clip-rsicd-v2')

    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError('Failed to open the image from URL.') from e

    formatted_texts = [f'a photo of a {label}' for label in labels]
    inputs = processor(text=formatted_texts, images=image, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze().tolist()

    return dict(zip(labels, probs))

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    image_url = 'https://example.com/city_park_image.jpg'
    labels = ['residential area', 'playground', 'stadium', 'forest', 'airport']

    # Test case 1: Valid image URL and labels
    print("Testing case [1/1] started.")
    result = classify_image(image_url, labels)
    assert isinstance(result, dict) and len(result) == len(labels), f"Test case [1/1] failed: Expected a dictionary with {len(labels)} items, got {len(result)} or not a dict."
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image()