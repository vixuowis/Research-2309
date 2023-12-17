# requirements_file --------------------

!pip install -U diffusers PIL

# function_import --------------------

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_image_from_text(description: str) -> Image:
    """
    Generates an image from a textual description using a pre-trained model.

    Args:
        description (str): The textual description from which to generate the image.

    Returns:
        An image generated based on the textual description.

    Raises:
        ValueError: If the description is empty.
    """
    if description == '':
        raise ValueError('The description cannot be empty.')
    model = 'CompVis/stable-diffusion-v1-4'
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
    pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)
    return pipe(description).images[0]

# test_function_code --------------------

def test_generate_image_from_text():
    print("Testing started.")
    # Test case 1: Valid description
    print("Testing case [1/1] started.")
    description = 'A futuristic city skyline at sunset.'
    assert isinstance(generate_image_from_text(description), Image), "Test case [1/1] failed: The result should be an image."
    print("Testing case [1/1] finished.")
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_from_text()