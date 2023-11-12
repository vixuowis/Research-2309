# function_import --------------------

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_image_from_text(text_description: str):
    '''
    Generate an image from a textual description using a pre-trained model.

    Args:
        text_description (str): The textual description to generate the image from.

    Returns:
        generated_image: The generated image from the textual description.
    '''
    model = 'CompVis/stable-diffusion-v1-4'
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
    pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)
    generated_image = pipe(text_description).images[0]
    return generated_image

# test_function_code --------------------

def test_generate_image_from_text():
    '''
    Test the function generate_image_from_text.
    '''
    text_description1 = 'A beautiful sunset over the ocean.'
    text_description2 = 'A cat sitting on a tree branch.'
    text_description3 = 'A group of people playing soccer in a park.'
    assert isinstance(generate_image_from_text(text_description1), type(None))
    assert isinstance(generate_image_from_text(text_description2), type(None))
    assert isinstance(generate_image_from_text(text_description3), type(None))
    print('All Tests Passed')

# call_test_function_code --------------------

test_generate_image_from_text()