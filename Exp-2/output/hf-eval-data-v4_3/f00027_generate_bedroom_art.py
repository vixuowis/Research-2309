# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_bedroom_art(model_name):
    """
    Generates an image of bedroom art using a diffusion model.

    Args:
        model_name (str): The name of the pretrained model to use.

    Returns:
        PIL.Image: The generated image of bedroom art.

    Raises:
        ValueError: If the model name is empty or None.
    """
    if not model_name:
        raise ValueError('Model name cannot be empty.')
    pipeline = DDPMPipeline.from_pretrained(model_name)
    generated_image = pipeline().images[0]
    return generated_image

# test_function_code --------------------

def test_generate_bedroom_art():
    print('Testing started.')

    model_name = 'johnowhitaker/sd-class-wikiart-from-bedrooms'

    # Testing with valid model name
    print('Testing case [1/3] started.')
    generated_image = generate_bedroom_art(model_name)
    assert generated_image is not None, 'Test case [1/3] failed: The generated image should not be None.'

    # Testing with empty model name
    print('Testing case [2/3] started.')
    try:
        generate_bedroom_art('')
    except ValueError as e:
        assert str(e) == 'Model name cannot be empty.', 'Test case [2/3] failed: Expected ValueError for empty model name.'

    # Testing with None as model name
    print('Testing case [3/3] started.')
    try:
        generate_bedroom_art(None)
    except ValueError as e:
        assert str(e) == 'Model name cannot be empty.', 'Test case [3/3] failed: Expected ValueError for None as model name.'

    print('Testing finished.')

# call_test_function_line --------------------

test_generate_bedroom_art()