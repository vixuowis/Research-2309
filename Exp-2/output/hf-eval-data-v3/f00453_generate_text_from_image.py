# function_import --------------------

from transformers import Pix2StructForConditionalGeneration
import PIL.Image

# function_code --------------------

def generate_text_from_image(image_path):
    """
    Generate text description from an image using Pix2Struct model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The generated text description of the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    model = Pix2StructForConditionalGeneration.from_pretrained('google/pix2struct-chartqa-base')
    image = PIL.Image.open(image_path)
    generated_text = model.generate_text(image)
    return generated_text

# test_function_code --------------------

def test_generate_text_from_image():
    """
    Test the function generate_text_from_image.
    """
    # Test with a valid image path
    image_path = 'path_to_valid_image.jpg'
    assert isinstance(generate_text_from_image(image_path), str)

    # Test with an invalid image path
    try:
        generate_text_from_image('path_to_invalid_image.jpg')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Expected a FileNotFoundError.')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_text_from_image()