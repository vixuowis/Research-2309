import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler


def generate_image(prompt: str, save_path: str = 'generated_image.png'):
    """
    This function generates an image based on the provided text prompt using the StableDiffusionPipeline model.
    
    Args:
    prompt (str): The text prompt based on which the image will be generated.
    save_path (str): The path where the generated image will be saved. Default is 'generated_image.png'.
    
    Returns:
    None
    """
    # Define the model id
    model_id = 'stabilityai/stable-diffusion-2-1-base'
    
    # Load the scheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder='scheduler')
    
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    
    # Move the model to GPU if available
    pipe = pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate the image
    image = pipe(prompt).images[0]
    
    # Save the image
    image.save(save_path)