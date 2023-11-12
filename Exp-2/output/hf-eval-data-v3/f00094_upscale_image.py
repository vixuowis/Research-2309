# function_import --------------------

from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch
import os

# function_code --------------------

def upscale_image(prompt: str, seed: int = 33):
    '''
    Upscale a low-resolution image of a movie character to a high-resolution image.

    Args:
        prompt (str): Text prompt describing the desired image.
        seed (int, optional): Seed for the random number generator. Defaults to 33.

    Returns:
        str: Path to the saved high-resolution image.
    '''
    pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
    pipeline.to('cuda')
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained('stabilityai/sd-x2-latent-upscaler', torch_dtype=torch.float16)
    upscaler.to('cuda')
    generator = torch.manual_seed(seed)
    low_res_latents = pipeline(prompt, generator=generator, output_type='latent').images
    upscaled_image = upscaler(prompt=prompt, image=low_res_latents, num_inference_steps=20, guidance_scale=0, generator=generator).images[0]
    image_path = 'movie_character_high_resolution.png'
    upscaled_image.save(image_path)
    return image_path

# test_function_code --------------------

def test_upscale_image():
    '''
    Test the upscale_image function.
    '''
    image_path = upscale_image('a photo of a movie character')
    assert os.path.exists(image_path), 'The image file does not exist.'
    assert os.path.getsize(image_path) > 0, 'The image file is empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_upscale_image()