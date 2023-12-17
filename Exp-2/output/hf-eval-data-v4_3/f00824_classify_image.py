# requirements_file --------------------

import subprocess

requirements = ["transformers", "pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    """
    Classify an image from a given URL using the pretrained ViT model.

    Args:
        image_url (str): The URL of the image to classify.

    Returns:
        torch.Tensor: The predicted features of the image.

    Raises:
        IOError: If the image cannot be opened from the URL.
        ValueError: If the image format is not supported.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except (requests.HTTPError, requests.ConnectionError) as e:
        raise IOError(f'Unable to open image from URL: {e}')
  
    if image.mode not in ['RGB', 'RGBA']:
        raise ValueError('Image format not supported. Expected RGB or RGBA.')

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'  # Example image URL

    # Test case 1: Valid image URL
    print("Testing case [1/1] started.")
    try:
        result = classify_image(test_image_url)
        assert result is not None, "Test case [1/1] failed: No result returned"
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image()