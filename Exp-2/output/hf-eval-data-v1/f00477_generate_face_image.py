from diffusers import DiffusionPipeline
import numpy as np


def generate_face_image(model_id: str = 'google/ncsnpp-celebahq-256', save_path: str = 'generated_face.png'):
    """
    This function generates a high-resolution image of a human face using a pre-trained model from Hugging Face Transformers.
    The generated image is saved to the specified path.

    Args:
        model_id (str): The ID of the pre-trained model to use for image generation. Default is 'google/ncsnpp-celebahq-256'.
        save_path (str): The path where the generated image will be saved. Default is 'generated_face.png'.
    """
    # Load the pre-trained model
    sde_ve = DiffusionPipeline.from_pretrained(model_id)

    # Generate a new image
    image = sde_ve()[0]

    # Save the generated image to a file
    image.save(save_path)