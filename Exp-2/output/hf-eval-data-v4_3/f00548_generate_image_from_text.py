# requirements_file --------------------

import subprocess

requirements = ["diffusers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_image_from_text(prompt: str, model_id: str = 'CompVis/stable-diffusion-v1-4', output_path: str = 'output.png'):
    """
    Generate an image from a text prompt using Stable Diffusion model.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str): The model ID for the Stable Diffusion model. Defaults to 'CompVis/stable-diffusion-v1-4'.
        output_path (str): The file path to save the generated image. Defaults to 'output.png'.

    Returns:
        str: The path to the saved image file.

    Raises:
        RuntimeError: If the device is not available for model inference.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU available for model inference')

    # Load the pretrained Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    # Generate the image
    image = pipe(prompt).images[0]
    image.save(output_path)

    return output_path

# test_function_code --------------------

def test_generate_image_from_text():
    print("Testing started.")

    # Test case: Generate an image with a known prompt
    print("Testing case [1/1] started.")
    output_path = generate_image_from_text('a serene lake at sunset', output_path='test_output.png')
    assert os.path.exists(output_path), f"Test case [1/1] failed: Image not saved to {output_path}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_from_text()