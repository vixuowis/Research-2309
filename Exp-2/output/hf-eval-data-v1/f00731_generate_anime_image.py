from diffusers import StableDiffusionPipeline
import torch


def generate_anime_image(prompt, negative_prompt):
    '''
    This function generates an anime-style image based on the provided text prompts using the 'dreamlike-art/dreamlike-anime-1.0' model.
    
    Parameters:
    prompt (str): A string describing the desired character appearance.
    negative_prompt (str): A string for features that should be excluded from the generated image.
    
    Returns:
    str: The path to the generated image.
    '''
    model_id = 'dreamlike-art/dreamlike-anime-1.0'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    image = pipe(prompt, negative_prompt=negative_prompt).images[0]
    image_path = './result.jpg'
    image.save(image_path)
    return image_path