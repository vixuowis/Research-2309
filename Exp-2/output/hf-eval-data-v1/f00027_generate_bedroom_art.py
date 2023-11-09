from diffusers import DDPMPipeline
from PIL import Image


def generate_bedroom_art():
    """
    This function generates a new image based on the online database of bedroom art.
    The function uses the DDPMPipeline class from the 'diffusers' library and the pretrained model 'johnowhitaker/sd-class-wikiart-from-bedrooms'.
    The generated image is saved as 'generated_bedroom_art.png'.
    """
    # Load the pretrained model
    pipeline = DDPMPipeline.from_pretrained('johnowhitaker/sd-class-wikiart-from-bedrooms')
    
    # Generate a new image
    generated_image = pipeline().images[0]
    
    # Save the generated image
    generated_image.save('generated_bedroom_art.png')
    
    return Image.open('generated_bedroom_art.png')