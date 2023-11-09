from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline


def load_text_to_image_model():
    '''
    This function loads a pre-trained model capable of converting text to images.
    The model is loaded from Hugging Face's model hub.
    '''
    # Specify the model name
    model = 'CompVis/stable-diffusion-v1-4'
    
    # Load the VAE (Variational Autoencoder) model
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
    
    # Create the Stable Diffusion Pipeline with the loaded VAE model
    pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)
    
    return pipe