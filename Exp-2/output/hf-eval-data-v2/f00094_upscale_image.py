# function_import --------------------

from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch

# function_code --------------------

def upscale_image(prompt: str, seed: int = 33):
    """
    Upscale a low-resolution image of a movie character to a high-resolution image.

    Args:
        prompt: A text prompt describing the desired image.
        seed: A seed for the random number generator. Default is 33.

    Returns:
        None. The function saves the upscaled high-resolution image as a .png file.
    """
    pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
    pipeline.to('cuda')
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained('stabilityai/sd-x2-latent-upscaler', torch_dtype=torch.float16)
    upscaler.to('cuda')
    generator = torch.manual_seed(seed)
    low_res_latents = pipeline(prompt, generator=generator, output_type='latent').images
    upscaled_image = upscaler(prompt=prompt, image=low_res_latents, num_inference_steps=20, guidance_scale=0, generator=generator).images[0]
    upscaled_image.save('movie_character_high_resolution.png')

# test_function_code --------------------

def test_upscale_image():
    """
    Test the upscale_image function.

    The function does not return a value, so the test will pass if the function runs without raising an exception.
    """
    upscale_image('a photo of a movie character')

# call_test_function_code --------------------

test_upscale_image()