# function_import --------------------

import os
from diffusers import DDPMPipeline

# function_code --------------------

def generate_bedroom_image(model_name: str, output_file: str) -> None:
    """
    Generate a realistic bedroom interior image using a pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model to use for image generation.
        output_file (str): The path where the generated image will be saved.

    Returns:
        None

    Raises:
        ValueError: If the model_name is not valid.
        FileNotFoundError: If the output_file path is not valid.
    """
    ddpm = DDPMPipeline.from_pretrained(model_name)
    image = ddpm().images[0]
    image.save(output_file)

# test_function_code --------------------

def test_generate_bedroom_image():
    """Tests for the `generate_bedroom_image` function"""
    # Test with valid inputs
    generate_bedroom_image('google/ddpm-bedroom-256', 'test_ddpm_generated_bedroom.png')
    assert os.path.exists('test_ddpm_generated_bedroom.png'), 'Test image not found.'
    # Clean up
    os.remove('test_ddpm_generated_bedroom.png')
    # Test with invalid model name
    try:
        generate_bedroom_image('invalid_model', 'test_ddpm_generated_bedroom.png')
    except ValueError:
        pass
    else:
        assert False, 'Expected a ValueError with invalid model name.'
    # Test with invalid output file path
    try:
        generate_bedroom_image('google/ddpm-bedroom-256', '/invalid/path/test_ddpm_generated_bedroom.png')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError with invalid output file path.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_bedroom_image()