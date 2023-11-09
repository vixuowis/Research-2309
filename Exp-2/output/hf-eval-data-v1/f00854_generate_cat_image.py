from diffusers import DDPMPipeline


def generate_cat_image(model_id: str = 'google/ddpm-ema-cat-256') -> None:
    """
    Generate a cat image using a pre-trained model from Hugging Face Transformers.

    Args:
        model_id (str): The identifier of the pre-trained model. Default is 'google/ddpm-ema-cat-256'.

    Returns:
        None. The function saves the generated image to the file 'ddpm_generated_cat_image.png'.
    """
    ddpm = DDPMPipeline.from_pretrained(model_id)
    image = ddpm().images[0]
    image.save('ddpm_generated_cat_image.png')