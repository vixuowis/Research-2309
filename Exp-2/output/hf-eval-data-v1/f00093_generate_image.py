import torch
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline

def generate_image(prompt, negative_prompt, strength=0.7):
    '''
    This function generates an image based on the given text prompt and negative prompt using the StableDiffusionDepth2ImgPipeline from Hugging Face.
    
    Parameters:
    prompt (str): The text prompt to generate the image.
    negative_prompt (str): The negative text prompt to avoid certain features in the image.
    strength (float, optional): The strength of the prompt effect on the generated image. Default is 0.7.
    
    Returns:
    PIL.Image: The generated image.
    '''
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth',
        torch_dtype=torch.float16,
    ).to('cuda')
    
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, strength=strength).images[0]
    return image