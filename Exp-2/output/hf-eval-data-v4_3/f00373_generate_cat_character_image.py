# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cat_character_image(model_id: str) -> None:
    """
    Generates a cartoon cat character image using a pretrained DDPM model.

    Args:
        model_id (str): The model ID of the pretrained DDPM to use for image generation.

    Returns:
        None: Saves the generated image to a file.

    Raises:
        ValueError: If the provided model ID is not a string.
        RuntimeError: If the model fails to load or generate the image.
    """
    if not isinstance(model_id, str):
        raise ValueError('model_id must be a string')

    try:
        ddpm = DDPMPipeline.from_pretrained(model_id)
        image = ddpm().images[0]
        image.save('cat_character_image.png')
    except Exception as e:
        raise RuntimeError(f'Image generation failed: {e}')

# test_function_code --------------------

def test_generate_cat_character_image():
    print("Testing started.")

    # Test case 1: Valid model ID
    print("Testing case [1/3] started.")
    try:
        generate_cat_character_image('test_model_id')
        print("Test case [1/3] passed.")
    except Exception as e:
        print(f"Test case [1/3] failed: {e}")

    # Test case 2: Invalid model ID (non-string)
    print("Testing case [2/3] started.")
    try:
        generate_cat_character_image(None)
        print("Test case [2/3] failed: Did not raise ValueError.")
    except ValueError as e:
        print("Test case [2/3] passed.")
    except Exception as e:
        print(f"Test case [2/3] failed: {e}")

    # Test case 3: Empty model ID
    print("Testing case [3/3] started.")
    try:
        generate_cat_character_image('')
        print("Test case [3/3] passed.")
    except Exception as e:
        print(f"Test case [3/3] failed: {e}")

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_cat_character_image()