# requirements_file --------------------

import subprocess

requirements = ["transformers", "pillow"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import Swin2SRForImageSuperResolution
from PIL import Image

# function_code --------------------

def upscale_image(image_path):
    """
    Upscale the image to twice its original size using the Swin2SR model.

    Args:
        image_path (str): The path to the image file to upscale.

    Returns:
        Image: The upscaled image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If any other unexpected error occurs during processing.
    """
    try:
        # Load the image
        original_image = Image.open(image_path)
        # Initialize the model
        model = Swin2SRForImageSuperResolution.from_pretrained('caidas/swin2sr-classical-sr-x2-64')
        # Perform the upscale
        upscaled_image = model(original_image)
        return upscaled_image
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"Image file not found: {image_path}") from fnf_error
    except Exception as e:
        raise Exception("An error occurred while processing the image.") from e

# test_function_code --------------------

def test_upscale_image():
    print("Testing started.")
    image_path = 'image_path.jpg'  # Path to a sample image
    
    # Test case 1: Upscale a valid image
    print("Testing case [1/2] started.")
    try:
        upscaled_image = upscale_image(image_path)
        assert upscaled_image is not None, "Test case [1/2] failed: upscaled_image is None"
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"

    # Test case 2: Upscale a non-existent image
    print("Testing case [2/2] started.")
    try:
        upscale_image('non_existent_image.jpg')
        assert False, "Test case [2/2] failed: FileNotFoundError not raised for non-existent image"
    except FileNotFoundError:
        pass  # Expected
    except Exception as e:
        assert False, f"Test case [2/2] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_upscale_image()