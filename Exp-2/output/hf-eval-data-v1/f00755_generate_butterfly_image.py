from diffusers import DDPMPipeline


def generate_butterfly_image(model_id: str = 'clp/sd-class-butterflies-32') -> None:
    """
    Generate a butterfly image using a pre-trained model from Hugging Face Transformers.

    Args:
        model_id (str): The identifier of the pre-trained model. Default is 'clp/sd-class-butterflies-32'.

    Returns:
        None. The function saves the generated image to the current directory.
    """
    pipeline = DDPMPipeline.from_pretrained(model_id)
    image = pipeline().images[0]
    image.save('cute_butterfly_image.png')