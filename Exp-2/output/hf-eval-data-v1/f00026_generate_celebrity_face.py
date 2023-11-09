import os
from diffusers import DDPMPipeline


def generate_celebrity_face(model_id='google/ddpm-ema-celebahq-256', save_path='generated_celebrity_face.png'):
    """
    This function generates a high-quality image of a celebrity face using the Denoising Diffusion Probabilistic Models (DDPM) from the Hugging Face Transformers.
    
    Parameters:
    model_id (str): The model id of the pretrained model. Default is 'google/ddpm-ema-celebahq-256'.
    save_path (str): The path where the generated image will be saved. Default is 'generated_celebrity_face.png'.
    
    Returns:
    None
    """
    # Check if the diffusers library is installed, if not, install it
    try:
        import diffusers
    except ImportError:
        os.system('pip install diffusers')
    
    # Create an instance of the DDPM pipeline with the given model id
    ddpm = DDPMPipeline.from_pretrained(model_id)
    
    # Generate a random image
    created_image = ddpm().images[0]
    
    # Save the generated image to the given path
    created_image.save(save_path)