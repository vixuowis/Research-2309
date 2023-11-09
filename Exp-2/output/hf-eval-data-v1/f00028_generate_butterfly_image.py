from diffusers import DDPMPipeline


def generate_butterfly_image():
    """
    This function uses the DDPMPipeline from the diffusers library to generate an image of a butterfly.
    The model used is 'clp/sd-class-butterflies-32' which is a diffusion model for unconditional image generation of cute butterflies.
    """
    # Load the pre-trained model
    pipeline = DDPMPipeline.from_pretrained('clp/sd-class-butterflies-32')
    
    # Generate the butterfly image
    image = pipeline().images[0]
    
    return image