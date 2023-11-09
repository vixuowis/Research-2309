from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline


def generate_image_from_text(text_description):
    '''
    This function generates an image from a textual description using the Hugging Face's StableDiffusionPipeline.
    
    Args:
    text_description (str): The textual description to generate the image from.
    
    Returns:
    generated_image: The generated image.
    '''
    # Load the pre-trained VAE (Variational Autoencoder) model
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
    
    # Use the pre-trained stable-diffusion-v1-4 model with the loaded VAE to create a text-to-image pipeline
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', vae=vae)
    
    # Generate the image from the textual description
    generated_image = pipe(text_description).images[0]
    
    return generated_image