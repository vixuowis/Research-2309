# requirements_file --------------------

import subprocess

requirements = ["requests", "Pillow", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# function_code --------------------

def generate_image_caption(image_url, context='product photography'):
    """
    Generate a descriptive caption for an image given a context.

    Args:
        image_url (str): The URL of the image for which to generate the caption.
        context (str): The context text that provides some background to the image. Defaults to 'product photography'.

    Returns:
        str: A descriptive caption generated for the image.

    Raises:
        ValueError: If the image URL is empty or invalid.
        Exception: If there is an issue in processing the image or generating the caption.
    """
    if not image_url:
        raise ValueError('The image URL must not be empty.')
    try:
        processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

        inputs = processor(raw_image, context, return_tensors='pt')
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        raise e

# test_function_code --------------------

def test_generate_image_caption():
    print("Testing started.")
    # Testing with a valid image URL and a specified context
    print("Testing case [1/3] started.")
    image_url = 'https://domain.com/sample-image.jpg'
    context = 'A sample product image.'
    caption = generate_image_caption(image_url, context)
    assert caption, f"Test case [1/3] failed: Expected a non-empty caption, got {caption}"

    # Testing with an invalid image URL
    print("Testing case [2/3] started.")
    invalid_image_url = 'https://domain.com/invalid-image.jpg'
    try:
        generate_image_caption(invalid_image_url)
        assert False, "Test case [2/3] failed: Expected a ValueError for invalid image URL"
    except ValueError:
        assert True

    # Testing with an empty image URL
    print("Testing case [3/3] started.")
    empty_image_url = ''
    try:
        generate_image_caption(empty_image_url)
        assert False, "Test case [3/3] failed: Expected a ValueError for empty image URL"
    except ValueError:
        assert True
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_caption()