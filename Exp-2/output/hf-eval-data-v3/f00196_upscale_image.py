# function_import --------------------

from transformers import Swin2SRForConditionalGeneration
from PIL import Image

# function_code --------------------

def upscale_image(low_res_image_path: str, high_res_image_path: str):
    """
    Upscale a low resolution image using the Swin2SRForConditionalGeneration model.

    Args:
        low_res_image_path (str): The path to the low resolution image.
        high_res_image_path (str): The path to save the upscaled image.

    Returns:
        None

    Raises:
        FileNotFoundError: If the low_res_image_path does not exist.
    """
    low_res_image = Image.open(low_res_image_path)
    model = Swin2SRForConditionalGeneration.from_pretrained('condef/Swin2SR-lightweight-x2-64')
    high_res_image = model.upscale_image(low_res_image)
    high_res_image.save(high_res_image_path)

# test_function_code --------------------

def test_upscale_image():
    """
    Test the upscale_image function.
    """
    # Test with a low resolution image
    upscale_image('low_res_image_path.jpg', 'high_res_image_path.jpg')
    assert Image.open('high_res_image_path.jpg').size > Image.open('low_res_image_path.jpg').size

    # Test with a different low resolution image
    upscale_image('low_res_image_path2.jpg', 'high_res_image_path2.jpg')
    assert Image.open('high_res_image_path2.jpg').size > Image.open('low_res_image_path2.jpg').size

    return 'All Tests Passed'

# call_test_function_code --------------------

test_upscale_image()