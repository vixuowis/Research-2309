from diffusers import DDPMPipeline

# Load the pretrained model
butterfly_generator = DDPMPipeline.from_pretrained('ocariz/butterfly_200')

def generate_butterfly_image():
    """
    Function to generate an image of a butterfly using a pretrained model.
    
    Args:
        None
    
    Returns:
        numpy.ndarray: The generated image of a butterfly.
    """
    # Generate the image
    butterfly_image = butterfly_generator().images[0]
    return butterfly_image