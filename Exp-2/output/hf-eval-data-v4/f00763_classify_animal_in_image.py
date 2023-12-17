# requirements_file --------------------

!pip install -U transformers Pillow requests

# function_import --------------------

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# function_code --------------------

def classify_animal_in_image(image_url: str) -> str:
    """
    Classify an animal in an image with a Chinese caption.

    Args:
        image_url (str): The URL of the image to classify.

    Returns:
        str: The classification result, 'cat' for 猫, 'dog' for 狗, or 'unknown'.
    """
    # Load pre-trained models
    model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')
    processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-base-patch16')

    # Obtain the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    texts = ['猫', '狗']  # 猫 for cat, 狗 for dog

    # Process the image and text inputs
    inputs = processor(images=image, text=texts, return_tensors='pt', padding=True)
    outputs = model(**inputs)

    # Calculate probabilities
    probs = outputs.logits_per_image.softmax(dim=1)
    highest_prob_idx = probs.argmax(dim=1).item()

    # Determine the classification result
    results = ['cat', 'dog']
    return results[highest_prob_idx] if highest_prob_idx < len(texts) else 'unknown'

# test_function_code --------------------

def test_classify_animal_in_image():
    print("Testing started.")
    # Test image URLs
    test_urls = {
        'cat_image': 'https://example.com/cat_image.jpg',
        'dog_image': 'https://example.com/dog_image.jpg',
        'unknown_image': 'https://example.com/unknown_image.jpg'
    }
    # Expected classifications
    expected = {
        'cat_image': 'cat',
        'dog_image': 'dog',
        'unknown_image': 'unknown'
    }

    for test_name, image_url in test_urls.items():
        print(f"Testing {test_name} started.")
        result = classify_animal_in_image(image_url)
        assert result == expected[test_name], f"Test {test_name} failed: expected {expected[test_name]}, got {result}"
        print(f"Testing {test_name} passed.")
    print("Testing finished.")