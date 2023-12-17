# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_image(model_id: str) -> 'Image':
    """
    Generate a cute butterfly image using a pre-trained DDPMPipeline model.

    Args:
        model_id: The ID of the pre-trained model to use for image generation.

    Returns:
        An Image object containing the generated image.

    Raises:
        ValueError: If the model_id is not a string or is empty.
        RuntimeError: If the image generation pipeline fails to execute.
    """
    if not model_id or not isinstance(model_id, str):
        raise ValueError('The model_id must be a non-empty string.')

    try:
        pipeline = DDPMPipeline.from_pretrained(model_id)
        image = pipeline().images[0]
        return image
    except Exception as e:
        raise RuntimeError('Image generation pipeline failed with: ' + str(e))

# test_function_code --------------------

import os

def test_generate_butterfly_image():
    print("Testing started.")
    model_id = 'clp/sd-class-butterflies-32'

    # Test case 1: Valid model_id
    print("Testing case [1/2] started.")
    image = generate_butterfly_image(model_id)
    assert isinstance(image, Image), f"Test case [1/2] failed: Expected an instance of Image, got {type(image)} instead."

    # Test case 2: Invalid model_id
    print("Testing case [2/2] started.")
    try:
        _ = generate_butterfly_image('')
        assert False, "Test case [2/2] failed: Expected a ValueError for empty model_id."
    except ValueError as e:
        assert str(e) == 'The model_id must be a non-empty string.', f"Test case [2/2] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_butterfly_image()