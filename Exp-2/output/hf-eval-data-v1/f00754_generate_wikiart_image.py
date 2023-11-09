from diffusers import DDPMPipeline


def generate_wikiart_image():
    """
    This function generates a new piece of art that resembles WikiArt images using the DDPMPipeline from the diffusers package.
    The function uses a pre-trained model 'johnowhitaker/sd-class-wikiart-from-bedrooms' which has been initialized from the 'google/ddpm-bedroom-256' model and further trained on the WikiArt dataset.
    
    Returns:
        image: A generated image that resembles images from the WikiArt dataset.
    """
    pipeline = DDPMPipeline.from_pretrained('johnowhitaker/sd-class-wikiart-from-bedrooms')
    image = pipeline().images[0]
    return image