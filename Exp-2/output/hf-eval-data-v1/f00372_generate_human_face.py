from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image


def generate_human_face():
    """
    This function generates a synthetic human face image using the pre-trained model 'google/ncsnpp-ffhq-256'.
    The generated image is then saved to a file named 'sde_ve_generated_image.png'.
    """
    # Load the pre-trained model
    sde_ve = DiffusionPipeline.from_pretrained('google/ncsnpp-ffhq-256')
    
    # Generate a synthetic human face image
    image = sde_ve().images[0]
    
    # Convert the image from numpy array to PIL image
    image = Image.fromarray(np.uint8(image * 255))
    
    # Save the image
    image.save('sde_ve_generated_image.png')