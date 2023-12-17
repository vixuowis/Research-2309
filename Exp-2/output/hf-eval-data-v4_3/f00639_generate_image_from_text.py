# requirements_file --------------------

import subprocess

requirements = ["diffusers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_image_from_text(prompt: str, model_id: str = 'dreamlike-art/dreamlike-photoreal-2.0') -> None:
    """Generates an image from the given text prompt using a pre-trained model.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str): The model ID of the pre-trained model to use. Default is 'dreamlike-art/dreamlike-photoreal-2.0'.

    Returns:
        None: Saves the generated image to a file.

    Raises:
        ValueError: If prompt is empty.
        RuntimeError: If there is an issue with the model loading or image generation.
    """
    if not prompt:
        raise ValueError('Prompt cannot be empty.')

    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
        generated_image = pipe(prompt).images[0]
        generated_image.save('result.png')
    except Exception as e:
        raise RuntimeError(f'Error generating image: {e}')

# test_function_code --------------------

def test_generate_image_from_text():
    print("Testing started.")
    # Test case 1: Valid prompt
    print("Testing case [1/1] started.")
    try:
        generate_image_from_text('astronaut playing guitar in space')
        print('Test case [1/1] passed.')
    except Exception as e:
        print(f'Test case [1/1] failed: {e}')
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_from_text()