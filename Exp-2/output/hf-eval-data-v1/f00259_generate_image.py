import torch
from diffusers import StableDiffusionPipeline


def generate_image(prompt: str, model_id: str = 'CompVis/stable-diffusion-v1-4', device: str = 'cuda') -> None:
    """
    This function generates an image based on the given text prompt using the StableDiffusionPipeline model.
    
    Parameters:
    prompt (str): The text prompt based on which the image is to be generated.
    model_id (str): The id of the model to be used for image generation. Default is 'CompVis/stable-diffusion-v1-4'.
    device (str): The device on which the model is to be loaded. Default is 'cuda'.
    
    Returns:
    None
    """
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    
    # Generate the image
    image = pipe(prompt).images[0]
    
    # Save the image
    image.save(f'{prompt.replace(" ", "_")}.png')