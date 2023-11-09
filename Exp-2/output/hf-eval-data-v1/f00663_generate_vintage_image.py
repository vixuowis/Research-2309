from diffusers import DDPMPipeline
import os


def generate_vintage_image(filename='vintage_magazine_cover.png'):
    """
    This function generates a vintage image using a pre-trained model from Hugging Face Transformers.
    The generated image is saved to the specified filename.
    
    Args:
    filename (str): The name of the file to save the generated image to. Defaults to 'vintage_magazine_cover.png'.
    
    Returns:
    None
    """
    # Load the pre-trained model
    pipeline = DDPMPipeline.from_pretrained('pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs')
    
    # Generate the image
    vintage_image = pipeline().images[0]
    
    # Save the image to the specified filename
    vintage_image.save(filename)
    
    print(f'Image saved to {os.path.abspath(filename)}')