# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

# function_code --------------------

def load_text_to_image_model():
    """
    Initializes and loads the text-to-image conversion model with fine-tuned VAE decoder.

    Args:
        None

    Returns:
        pipe (StableDiffusionPipeline): The loaded pipeline with the capability to generate images from text.

    Raises:
        OSError: If the model cannot be loaded.
    """
    try:
        model = 'CompVis/stable-diffusion-v1-4'
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
        pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)
        return pipe
    except Exception as e:
        raise OSError(f'Model could not be loaded: {str(e)}')

# test_function_code --------------------

def test_load_text_to_image_model():
    print("Testing started.")
    
    # Test case 1: Checking if the function returns a StableDiffusionPipeline instance
    print("Testing case [1/1] started.")
    try:
        pipe = load_text_to_image_model()
        assert isinstance(pipe, StableDiffusionPipeline), f"Test case [1/1] failed: The function did not return a StableDiffusionPipeline instance."
    except AssertionError as ae:
        print(str(ae))
    
    print("Testing finished.")

# call_test_function_line --------------------

test_load_text_to_image_model()