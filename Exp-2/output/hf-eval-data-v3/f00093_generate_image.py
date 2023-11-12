# function_import --------------------

import torch
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline

# function_code --------------------

def generate_image(prompt: str, negative_prompt: str, strength: float = 0.7) -> Image:
    """
    Generate an image based on the given text prompts using the StableDiffusionDepth2ImgPipeline model.

    Args:
        prompt (str): Text prompt to generate image.
        negative_prompt (str): Negative text prompt to avoid certain features.
        strength (float, optional): Strength of the prompt effect on the generated image. Defaults to 0.7.

    Returns:
        Image: The generated image.
    """
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth',
        torch_dtype=torch.float16,
    ).to('cuda')

    image = pipe(prompt=prompt, negative_prompt=negative_prompt, strength=strength).images[0]
    return image

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.
    """
    prompt = 'two tigers'
    negative_prompt = 'bad, deformed, ugly, bad anatomy'
    image = generate_image(prompt, negative_prompt)
    assert isinstance(image, Image.Image), 'The result should be an instance of PIL.Image.Image'

    prompt = 'a beautiful landscape'
    negative_prompt = 'dark, gloomy, scary'
    image = generate_image(prompt, negative_prompt)
    assert isinstance(image, Image.Image), 'The result should be an instance of PIL.Image.Image'

    prompt = 'a serene beach'
    negative_prompt = 'crowded, dirty, polluted'
    image = generate_image(prompt, negative_prompt)
    assert isinstance(image, Image.Image), 'The result should be an instance of PIL.Image.Image'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image()