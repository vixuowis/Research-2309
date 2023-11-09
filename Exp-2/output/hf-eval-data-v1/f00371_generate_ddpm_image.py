import torch
from diffusers import DDPMPipeline


def generate_ddpm_image(model_id: str = 'google/ddpm-church-256') -> None:
    """
    This function generates a high-quality image of a church using the Denoising Diffusion Probabilistic Models (DDPM) from the Hugging Face Transformers.
    The generated image is saved as 'ddpm_generated_image.png'.

    Args:
        model_id (str, optional): The model id of the pretrained model. Defaults to 'google/ddpm-church-256'.
    """
    # Load the pretrained model
    ddpm = DDPMPipeline.from_pretrained(model_id)

    # Generate the image
    image = ddpm().images[0]

    # Save the generated image
    image.save('ddpm_generated_image.png')