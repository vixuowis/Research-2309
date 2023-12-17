# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_image_from_description(description):
    '''
    Generate an image based on the given textual description.

    Parameters:
        description (str): A textual description of the desired image.

    Returns:
        Image: The generated image based on the description.
    '''
    model = 'CompVis/stable-diffusion-v1-4'
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
    pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)
    image = pipe(description)
    return image


# test_function_code --------------------

def test_generate_image_from_description():
    print("Testing started.")
    sample_description = 'A cozy room with a fireplace and a white cat sleeping on the sofa.'

    print("Generating image [1/1] started.")
    image = generate_image_from_description(sample_description)
    assert image is not None, f"Image generation failed."
    print("Image successfully generated.")
    print("Testing finished.")

# Run the test function
test_generate_image_from_description()
