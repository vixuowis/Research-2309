# requirements_file --------------------

import subprocess

requirements = ["transformers", "Pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url: str) -> str:
    """
    Classify an image as a cat or a dog using a pre-trained Vision Transformer model.

    Args:
        image_url: str - A URL to an image of a cat or a dog.

    Returns:
        str - The predicted class ('cat' or 'dog').

    Raises:
        ValueError: If the image URL is invalid or the image cannot be loaded.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError(f'Failed to load image from the URL: {image_url}') from e

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_categories = ['cat', 'dog']
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_categories[predicted_class_idx]

# test_function_code --------------------

def test_classify_image():
    print('Testing started.')
    # Sample image URLs
    sample_images = [
        'http://example.com/cat.jpg',
        'http://example.com/dog.jpg'
    ]

    # Test case 1: Classify cat image
    print('Testing case [1/2] started.')
    result = classify_image(sample_images[0])
    assert result == 'cat', f'Test case [1/2] failed: Expected cat, got {result}'

    # Test case 2: Classify dog image
    print('Testing case [2/2] started.')
    result = classify_image(sample_images[1])
    assert result == 'dog', f'Test case [2/2] failed: Expected dog, got {result}'

    print('Testing finished.')

# call_test_function_line --------------------

test_classify_image()