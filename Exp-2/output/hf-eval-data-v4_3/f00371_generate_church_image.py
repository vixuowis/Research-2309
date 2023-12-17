# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_church_image(model_id: str) -> Image:
    """
    Generate an image of a church using a specified DDPM model.

    Args:
        model_id: A string identifier for the pretrained DDPM model.

    Returns:
        A PIL Image object of the generated church image.

    Raises:
        ValueError: If the 'model_id' is not provided or an empty string.
    """
    if not model_id:
        raise ValueError("'model_id' must be a valid non-empty string.")
    ddpm = DDPMPipeline.from_pretrained(model_id)
    image = ddpm().images[0]
    return image

# test_function_code --------------------

def test_generate_church_image():
    print("Testing started.")
    model_id = 'google/ddpm-church-256'

    # Test case 1: Valid model_id
    print("Testing case [1/3] started.")
    image = generate_church_image(model_id)
    assert image is not None, "Test case [1/3] failed: The function did not return an image."

    # Test case 2: model_id is an empty string
    print("Testing case [2/3] started.")
    try:
        generate_church_image('')
        assert False, "Test case [2/3] failed: The function should raise a ValueError."
    except ValueError as e:
        assert str(e) == "'model_id' must be a valid non-empty string.", "Test case [2/3] failed: The function raised a different ValueError than expected."

    # Test case 3: model_id is None
    print("Testing case [3/3] started.")
    try:
        generate_church_image(None)
        assert False, "Test case [3/3] failed: The function should raise a ValueError."
    except ValueError as e:
        assert str(e) == "'model_id' must be a valid non-empty string.", "Test case [3/3] failed: The function raised a different ValueError than expected."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_church_image()