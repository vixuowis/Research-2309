import torch
from diffusers import DiffusionPipeline


def generate_image(model_id: str = 'CompVis/ldm-celebahq-256', num_inference_steps: int = 200) -> torch.Tensor:
    '''
    Function to generate high-quality images of faces using a pre-trained model from Hugging Face Transformers.
    
    Parameters:
    model_id (str): The id of the pre-trained model. Default is 'CompVis/ldm-celebahq-256'.
    num_inference_steps (int): The number of inference steps. Default is 200.
    
    Returns:
    torch.Tensor: The generated image.
    '''
    # Import the DiffusionPipeline class from the Python package 'diffusers' created by Hugging Face.
    # Use the from_pretrained method to load the pre-trained model that has been trained to generate high-resolution images of faces.
    pipeline = DiffusionPipeline.from_pretrained(model_id)
    
    # Generate a new high-quality image by specifying the number of inference steps.
    image = pipeline(num_inference_steps=num_inference_steps)
    
    return image[0]