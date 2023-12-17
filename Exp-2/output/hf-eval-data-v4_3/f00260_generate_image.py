# requirements_file --------------------

import subprocess

requirements = ["diffusers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_image(prompt: str) -> str:
    """
    Generates an image based on the provided text prompt using the anything-v4.0 model.

    Args:
        prompt (str): The text prompt used for image generation.

    Returns:
        str: The file path of the saved image.

    Raises:
        RuntimeError: If the model fails to generate the image.
    """
    model_id = 'andite/anything-v4.0'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
    generated_image = pipe(prompt).images[0]
    file_path = './anime_girl_guitar.png'
    generated_image.save(file_path)
    return file_path

# test_function_code --------------------

def test_generate_image():
    print("Testing started.")
    # Test case: Generate an image with the prompt 'anime-style girl with a guitar'.
    print("Testing case [1/1] started.")
    file_path = generate_image('anime-style girl with a guitar')
    assert os.path.exists(file_path), f"Test case [1/1] failed: Image file not found at {file_path}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image()