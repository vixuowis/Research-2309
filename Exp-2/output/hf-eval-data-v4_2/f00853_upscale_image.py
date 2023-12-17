# requirements_file --------------------

!pip install -U transformers torch Pillow requests

# function_import --------------------

from transformers import Swin2SRForConditionalGeneration
import torch
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def upscale_image(image_url: str) -> Image.Image:
    """
    Upscales a low-resolution image to twice its size using a pretrained Swin2SR model.

    Args:
        image_url: The URL of the low-resolution image to be upscaled.

    Returns:
        An Image.Image object representing the upscaled image.

    Raises:
        ValueError: If the image URL is not valid.
    """
    # Load the pretrained Swin2SR model
    model = Swin2SRForConditionalGeneration.from_pretrained('conde/Swin2SR-lightweight-x2-64')
    feature_extractor = model.feature_extractor

    # Download the image
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError('Image URL is not valid or cannot be accessed.')

    # Preprocess the image
    low_resolution_image = Image.open(BytesIO(response.content))
    input_tensor = feature_extractor(images=low_resolution_image, return_tensors='pt').input_ids

    # Generate the high resolution image with the model
    with torch.no_grad():
        upscaled_tensor = model.generate(input_tensor)

    # Postprocess the output
    upscaled_image = feature_extractor.decode(upscaled_tensor[0])

    return upscaled_image

# test_function_code --------------------

def test_upscale_image():
    print("Testing started.")
    test_url = 'https://example.com/low_res_image.jpg'  # Replace with an actual image URL

    # Test case 1: Upscale a valid low-resolution image
    print("Testing case [1/1] started.")
    try:
        result = upscale_image(test_url)
        assert isinstance(result, Image.Image), 'Result is not an image.'
        print("Test case [1/1] passed.")
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")
    print("Testing finished.")

# call_test_function_line --------------------

test_upscale_image()