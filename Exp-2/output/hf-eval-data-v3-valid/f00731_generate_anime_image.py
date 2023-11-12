# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch
import os

# function_code --------------------

def generate_anime_image(prompt: str, negative_prompt: str, save_path: str = './result.jpg'):
    '''
    Generate an anime image based on the given prompt and negative_prompt.

    Args:
        prompt (str): The description of the desired character appearance.
        negative_prompt (str): The features that should be excluded from the generated image.
        save_path (str, optional): The path to save the generated image. Defaults to './result.jpg'.

    Returns:
        None
    '''
    model_id = 'dreamlike-art/dreamlike-anime-1.0'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    image = pipe(prompt, negative_prompt=negative_prompt).images[0]
    image.save(save_path)

# test_function_code --------------------

def test_generate_anime_image():
    '''
    Test the function generate_anime_image.
    '''
    prompt = 'anime, masterpiece, high quality, 1girl, solo, long hair, looking at viewer, blush, smile, bangs, blue eyes, skirt, medium breasts, iridescent, gradient, colorful'
    negative_prompt = 'simple background, duplicate, retro style, low quality, lowest quality, 1980s, 1990s, 2000s, 2005 2006 2007 2008 2009 2010 2011 2012 2013, bad anatomy, bad proportions, extra digits, lowres, username, artist name, error, duplicate, watermark, signature, text, extra digit, fewer digits, worst quality, jpeg artifacts, blurry'
    generate_anime_image(prompt, negative_prompt)
    assert os.path.exists('./result.jpg'), 'Image not generated'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_anime_image()