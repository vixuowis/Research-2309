# requirements_file --------------------

!pip install -U requests PIL transformers

# function_import --------------------

from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

# function_code --------------------

def generate_image_caption(image_url, context):
    """
    Generate a descriptive caption for an image with context.

    Args:
    image_url (str): URL of the image to caption.
    context (str): Contextual information about the image.

    Returns:
    str: A descriptive caption generated for the image.
    """
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

    response = requests.get(image_url, stream=True)
    image = Image.open(response.raw).convert('RGB')

    inputs = processor(image, context, return_tensors='pt')
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

# test_function_code --------------------

def test_generate_image_caption():
    print("Testing generate_image_caption function.")
    sample_image_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    sample_context = 'product photography'

    # Test case 1
    print("Testing case [1/1] started.")
    caption = generate_image_caption(sample_image_url, sample_context)
    assert caption is not None and len(caption) > 0, "Test case [1/1] failed: Empty caption."
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_generate_image_caption()