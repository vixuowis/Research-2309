from diffusers import DDPMPipeline


def generate_butterfly_image():
    """
    This function generates images of cute butterflies using the 'myunus1/diffmodels_galaxies_scratchbook' model.

    Returns:
        PIL.Image: Generated image of a butterfly.
    """
    pipeline = DDPMPipeline.from_pretrained('myunus1/diffmodels_galaxies_scratchbook')
    generated_data = pipeline()
    image = generated_data.images[0]
    return image