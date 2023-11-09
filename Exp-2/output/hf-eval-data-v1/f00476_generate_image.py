import torch
from diffusers import DDPMPipeline


def generate_image(model_id: str = 'google/ddpm-church-256') -> None:
    '''
    This function generates an image using the DDPMPipeline from the diffusers package.
    The model used is specified by the model_id parameter.
    The generated image is saved in the current directory.
    
    Parameters:
    model_id (str): The id of the model to use for image generation. Default is 'google/ddpm-church-256'.
    
    Returns:
    None
    '''
    # Load the pretrained model
    ddpm = DDPMPipeline.from_pretrained(model_id)
    
    # Generate an image
    image = ddpm().images[0]
    
    # Save the generated image
    image.save('ddpm_generated_image.png')