from diffusers import StableDiffusionPipeline
import torch

def generate_anime_style_image(prompt):
    '''
    Function to generate an anime-style image based on a given prompt using the 'andite/anything-v4.0' model.
    
    Parameters:
    prompt (str): The prompt based on which the image is to be generated.
    
    Returns:
    None. The generated image is saved in the current directory.
    '''
    model_id = 'andite/anything-v4.0'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    generated_image = pipe(prompt).images[0]
    generated_image.save('./anime_girl_guitar.png')