# requirements_file --------------------

!pip install -U requests PIL transformers

# function_import --------------------

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def generate_park_image_description(image_path_or_url):
    """
    Generate a description for a park image using BLIP image captioning model.

    Parameters:
    - image_path_or_url: The path or URL to the image of the park.

    Returns:
    - caption: A string containing the generated description of the park image.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

    if image_path_or_url.startswith('http'):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')

    inputs = processor(image, return_tensors='pt')
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# test_function_code --------------------

def test_generate_park_image_description():
    print("Testing started.")

    # Test case: URL image
    print("Testing with URL image started.")
    url_image = 'https://example.com/sample_park_image.jpg'
    url_caption = generate_park_image_description(url_image)
    assert isinstance(url_caption, str), "URL image test failed: The function must return a string description."
    print("Testing with URL image finished.")

    # Test case: local image file path
    print("Testing with local image file started.")
    local_image = 'path_to_local_park_image.jpg'
    local_caption = generate_park_image_description(local_image)
    assert isinstance(local_caption, str), "Local image file test failed: The function must return a string description."
    print("Testing with local image file finished.")

    print("Testing finished.")

# Run the test function
test_generate_park_image_description()