# requirements_file --------------------

!pip install -U diffusers Pillow

# function_import --------------------

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_image_from_text(description):
    """
    Generate an image based on the provided textual description using the
    pre-trained Stable Diffusion model.

    :param description: str - The textual description to generate an image from.
    :return: PIL.Image - The generated image.
    """
    model = 'CompVis/stable-diffusion-v1-4'
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
    pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)
    return pipe(description).images[0]

# test_function_code --------------------

def test_generate_image_from_text():
    print("Testing started.")

    # Setting up a dummy description
    description = "A tranquil beach at sunset"

    # Test case 1: Check if the function returns an image
    print("Testing case [1/1] started.")
    result = generate_image_from_text(description)
    assert isinstance(result, Image.Image), f"Test case [1/1] failed: Expected PIL.Image, got {type(result)}"
    print("Testing finished.")