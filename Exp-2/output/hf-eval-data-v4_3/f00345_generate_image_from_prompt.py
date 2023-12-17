# requirements_file --------------------

import subprocess

requirements = ["diffusers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_image_from_prompt(prompt: str) -> None:
    """
    Generates an image from the provided text prompt using the StableDiffusionPipeline.

    Args:
        prompt (str): A descriptive text of the image to be generated.

    Returns:
        None: The function saves the generated image locally.

    Raises:
        ValueError: If the prompt is empty.
        RuntimeError: If there is an issue with model loading or image generation.
    """
    if not prompt:
        raise ValueError('The prompt cannot be empty.')

    model_id = 'prompthero/openjourney'
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    except Exception as e:
        raise RuntimeError(f'Failed to load the model: {e}')

    try:
        pipe = pipe.to('cuda')
        image = pipe(prompt).images[0]
        image.save('./generated_image.png')
    except Exception as e:
        raise RuntimeError(f'Failed to generate the image: {e}')

# test_function_code --------------------

def test_generate_image_from_prompt():
    print("Testing started.")

    # Test case 1: valid prompt
    print("Testing case [1/2] started.")
    try:
        generate_image_from_prompt('A vintage sports car racing through a desert landscape during sunset')
    except Exception as e:
        assert False, f'Test case [1/2] failed: {e}'

    # Test case 2: empty prompt
    print("Testing case [2/2] started.")
    try:
        generate_image_from_prompt('')
        assert False, 'Test case [2/2] failed: No ValueError raised for empty prompt.'
    except ValueError:
        pass  # Expected
    except Exception as e:
        assert False, f'Test case [2/2] failed: {e}'

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_from_prompt()