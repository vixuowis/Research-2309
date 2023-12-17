# requirements_file --------------------

!pip install -U transformers pillow

# function_import --------------------

from transformers import Swin2SRForConditionalGeneration
from PIL import Image

# function_code --------------------

def sharpen_drone_images(image_path):
    """
    Sharpens the image captured from a drone.

    Args:
        image_path (str): The file path to the image to be sharpened.

    Returns:
        PIL.Image.Image: The sharpened image.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        Exception: If the image could not be processed.
    """
    # Load the image
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise FileNotFoundError(f'Unable to open image: {e}')

    # Load the super-resolution model
    model = Swin2SRForConditionalGeneration.from_pretrained('condef/Swin2SR-lightweight-x2-64')

    # Preprocess the image for the model
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Process the image using the model
    try:
        outputs = model(**inputs)
    except Exception as e:
        raise Exception(f'Unable to process image: {e}')

    # Post-process the output and return the sharpened image
    # Implementation depends on the library used and how the model outputs results
    # Assuming we have the function post_process_output for illustrative purposes
    sharpened_image = post_process_output(outputs)

    return sharpened_image

# test_function_code --------------------

def test_sharpen_drone_images():
    print("Testing started.")
    # Test with a valid image path
    print("Testing case [1/2] started.")
    sharpened_image = sharpen_drone_images('valid_image_path.jpg')
    assert isinstance(sharpened_image, Image.Image), f"Test case [1/2] failed: Expected PIL.Image.Image, got {type(sharpened_image)}"

    # Test with an invalid image path
    print("Testing case [2/2] started.")
    try:
        sharpen_drone_images('invalid_image_path.jpg')
        assert False, "Test case [2/2] failed: Expected FileNotFoundError"
    except FileNotFoundError:
        pass
    print("Testing finished.")

# call_test_function_line --------------------

test_sharpen_drone_images()