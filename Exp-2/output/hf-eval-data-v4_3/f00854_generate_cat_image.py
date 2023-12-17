# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cat_image(model_id: str) -> str:
    """
    Generates an image of a cat using the Denoising Diffusion Probabilistic Model.

    Args:
        model_id (str): A string representing the model ID of the pre-trained DDPM.

    Returns:
        str: The file path to the saved cat image.

    Raises:
        ValueError: If the model_id provided is incorrect or not available.
    """
    try:
        ddpm = DDPMPipeline.from_pretrained(model_id)
    except Exception as e:
        raise ValueError(f"Could not load the model with id {model_id}") from e

    image = ddpm().images[0]
    file_path = 'ddpm_generated_cat_image.png'
    image.save(file_path)
    return file_path

# test_function_code --------------------

def test_generate_cat_image():
    print("Testing started.")
    
    # Test case 1: Generating a cat image with a valid model_id
    print("Testing case [1/1] started.")
    model_id = 'google/ddpm-ema-cat-256'
    file_path = generate_cat_image(model_id)
    assert file_path == 'ddpm_generated_cat_image.png', f"Test case [1/1] failed: Expected 'ddpm_generated_cat_image.png', got {file_path}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_cat_image()