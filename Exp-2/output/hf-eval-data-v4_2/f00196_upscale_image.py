# requirements_file --------------------

!pip install -U transformers==4.19.2 Pillow==9.1.0

# function_import --------------------

from transformers import Swin2SRForConditionalGeneration
from PIL import Image

# function_code --------------------

def upscale_image(low_res_image_path: str, save_path: str) -> str:
    """
    Upscales a low resolution image to a higher resolution using a pre-trained model.

    Args:
        low_res_image_path: The file path to the low resolution image.
        save_path: The file path where the upscaled image will be saved.

    Returns:
        The file path to the saved high resolution image.

    Raises:
        FileNotFoundError: If the low resolution image file does not exist.
        Exception: If there are issues during the upscaling or saving process.
    """
    try:
        low_res_image = Image.open(low_res_image_path)
    except FileNotFoundError:
        raise
    model = Swin2SRForConditionalGeneration.from_pretrained('condef/Swin2SR-lightweight-x2-64')
    high_res_image = model.upscale_image(low_res_image)
    high_res_image.save(save_path)
    return save_path

# test_function_code --------------------

def test_upscale_image():
    print("Testing started.")
    # Assuming 'low_res_image_test.jpg' exists.
    low_res_image_path = 'low_res_image_test.jpg'
    save_path = 'high_res_image_test.jpg'

    print("Testing case [1/1] started.")
    try:
        upscaled_image_path = upscale_image(low_res_image_path, save_path)
        assert os.path.exists(upscaled_image_path), f"Test case [1/1] failed: Upscaled file not found at: {upscaled_image_path}"
    except FileNotFoundError as fnfe:
        assert False, f"Test case [1/1] failed: {str(fnfe)}"
    except Exception as e:
        assert False, f"Test case [1/1] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_upscale_image()