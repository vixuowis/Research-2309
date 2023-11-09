from diffusers import DDPMPipeline


def generate_butterfly_image():
    """
    This function generates an image of a butterfly using a pre-trained model from Hugging Face Transformers.
    The model is specifically trained for generating images of butterflies.
    
    Returns:
        generated_image: An image of a butterfly generated by the model.
    """
    # Load the pre-trained model
    pipeline = DDPMPipeline.from_pretrained('utyug1/sd-class-butterflies-32')
    
    # Generate a new butterfly image
    generated_image = pipeline().images[0]
    
    return generated_image