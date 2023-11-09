from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image


def generate_story_image(prompt):
    '''
    This function generates an image based on a text description using the StableDiffusionPipeline model.
    
    Args:
    prompt (str): The text description of the scene.
    
    Returns:
    str: The filename of the saved image.
    '''
    # Load the pre-trained model
    model_id = 'stabilityai/stable-diffusion-2-1'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    # Initialize the scheduler and move the pipeline to the GPU
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')
    
    # Generate the image
    generated_image = pipe(prompt).images[0]
    
    # Save the image
    filename = 'generated_image.png'
    generated_image.save(filename)
    
    return filename