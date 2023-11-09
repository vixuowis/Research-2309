# function_import --------------------

import torch
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline

# function_code --------------------

def generate_tiger_image(prompt: str, negative_prompt: str, strength: float) -> None:
    """
    Generate and save an image of tigers based on the given prompts using the StableDiffusionDepth2ImgPipeline model.

    Args:
        prompt (str): Text prompt to generate image.
        negative_prompt (str): Negative text prompt to avoid certain features.
        strength (float): Strength of the prompt effect on the generated image.

    Returns:
        None. The function saves the generated image to the current directory.
    """
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth',
        torch_dtype=torch.float16,
    ).to('cuda')

    image = pipe(prompt=prompt, negative_prompt=negative_prompt, strength=strength).images[0]
    image.save('generated_tigers_image.png')

# test_function_code --------------------

def test_generate_tiger_image():
    """
    Test the generate_tiger_image function.
    """
    generate_tiger_image('two tigers', 'bad, deformed, ugly, bad anatomy', 0.7)
    assert Image.open('generated_tigers_image.png') is not None

# call_test_function_code --------------------

test_generate_tiger_image()