from diffusers import DDPMPipeline


def generate_classical_image():
    """
    This function generates a classical image using a pretrained diffusion model.
    The model is trained to recreate the style of classical images.
    Once the model is loaded, a new image is generated by simply calling the model.
    The generated image will be available in the model's output.
    """
    # Load the pretrained diffusion model
    pipeline = DDPMPipeline.from_pretrained('johnowhitaker/sd-class-wikiart-from-bedrooms')
    
    # Generate a new image
    generated_image = pipeline.generate_image()
    
    return generated_image