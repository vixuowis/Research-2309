# requirements_file --------------------

import subprocess

requirements = ["diffusers", "transformers", "accelerate", "scipy", "safetensors"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image

# function_code --------------------

def generate_storybook_image(prompt: str, output_filename: str) -> str:
    """
    Generates an image from a text description for a children's storybook using Stable Diffusion model.
    
    Args:
        prompt (str): The text description of the scene to generate.
        output_filename (str): The filename to save the generated image.

    Returns:
        str: The filename of the saved image.

    Raises:
        ValueError: If the prompt is empty or None.
        IOError: If there is an error saving the image.
    """
    if not prompt:
        raise ValueError('The prompt is empty or None')

    model_id = 'stabilityai/stable-diffusion-2-1'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')

    generated_image = pipe(prompt).images[0]
    generated_image.save(output_filename)
    return output_filename

# test_function_code --------------------

def test_generate_storybook_image():
    print("Testing started.")
    prompt = "a fantasy castle in the sky"
    output_filename = "fantasy_castle.png"

    # Testing case 1: Check if the function returns the correct output filename
    print("Testing case [1/1] started.")
    result = generate_storybook_image(prompt, output_filename)
    assert result == output_filename, f"Test case [1/1] failed: Expected {output_filename}, got {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_storybook_image()