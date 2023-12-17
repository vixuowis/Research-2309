# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL

# function_code --------------------

def generate_images_from_text(description, model_path='CompVis/stable-diffusion-v1-4', vae_path='stabilityai/sd-vae-ft-ema'):
    # Load the Variational Autoencoder (VAE) model
    vae = AutoencoderKL.from_pretrained(vae_path)
    # Create the Stable Diffusion Pipeline with the provided model and VAE
    pipe = StableDiffusionPipeline.from_pretrained(model_path, vae=vae)
    # Generate images from the provided textual description
    images = pipe(description).images
    # Return generated images
    return images

# test_function_code --------------------

def test_generate_images_from_text():
    print("Testing generate_images_from_text function...")
    test_description = 'A clear night sky with twinkling stars and a bright full moon.'
    images = generate_images_from_text(test_description)
    assert len(images) > 0, f'No images generated for description: {test_description}'
    print('Test passed. Images generated successfully.')

# Running the test function
test_generate_images_from_text()