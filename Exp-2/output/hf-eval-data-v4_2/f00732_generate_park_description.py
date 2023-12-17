# requirements_file --------------------

!pip install -U torchvision transformers requests pillow

# function_import --------------------

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# function_code --------------------

def generate_park_description(img_url):
    """
    Generates a description of a park image for promotional purposes.

    Args:
        img_url (str): The URL or local path of the park image to describe.

    Returns:
        str: A textual description of the park image.

    Raises:
        Exception: If the image cannot be processed or the model cannot generate a description.
    """
    try:
        processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        inputs = processor(raw_image, return_tensors='pt')
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        raise Exception(f'Failed to generate description due to {e}')

# test_function_code --------------------

def test_generate_park_description():
    print("Testing started.")
    img_url = 'path_or_url_to_your_park_image.jpg'  # Replace with an actual image URL or local path

    # Testing case 1: Image URL
    print("Testing case [1/1] started.")
    description = generate_park_description(img_url)
    assert isinstance(description, str), f"Test case [1/1] failed: Description is not a string."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_park_description()