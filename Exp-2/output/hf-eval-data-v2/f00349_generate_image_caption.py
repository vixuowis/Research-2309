# function_import --------------------

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# function_code --------------------

def generate_image_caption(image_path: str, text: str = 'product photography') -> str:
    """
    Generate a descriptive caption for a given image using the Salesforce/blip-image-captioning-base model.

    Args:
        image_path (str): The path to the image for which to generate a caption.
        text (str, optional): A short text that provides some context to the photograph. Defaults to 'product photography'.

    Returns:
        str: The generated caption for the image.
    """
    # Initialize the processor and model
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

    # Open the image
    image = Image.open(image_path)

    # Process the image and text
    inputs = processor(image, text, return_tensors='pt')

    # Generate the caption
    out = model.generate(**inputs)

    # Decode and return the caption
    return processor.decode(out[0], skip_special_tokens=True)

# test_function_code --------------------

def test_generate_image_caption():
    """
    Test the generate_image_caption function.
    """
    # Define the image path and text
    image_path = 'test_image.jpg'
    text = 'test photography'

    # Generate the caption
    caption = generate_image_caption(image_path, text)

    # Assert that the caption is a string
    assert isinstance(caption, str)

    # Assert that the caption is not empty
    assert len(caption) > 0

# call_test_function_code --------------------

test_generate_image_caption()