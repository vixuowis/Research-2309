# requirements_file --------------------

!pip install -U git+https://github.com/huggingface/diffusers.git transformers accelerate scipy safetensors

# function_import --------------------

from diffusers import StableDiffusionPipeline, StableDiffusionLatentUpscalePipeline
import torch

# function_code --------------------

def upscale_movie_character_images(prompt, seed=33):
    """
    Upscale low-resolution images of movie characters to high-resolution using pretrained models.

    Args:
        prompt (str): Text prompt describing the movie character image to upscale.
        seed (int, optional): Seed value for the torch generator to ensure reproducibility. Defaults to 33.

    Returns:
        Image: The upscaled high-resolution image.

    Raises:
        RuntimeError: If there is an error in the upscaling process.
    """
    torch.manual_seed(seed)
    base_pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16).to('cuda')
    upscaler_pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained('stabilityai/sd-x2-latent-upscaler', torch_dtype=torch.float16).to('cuda')

    # Generate low-resolution latents
    low_res_latents = base_pipeline(prompt, generator=torch.Generator(), output_type='latent').images

    # Generate upscaled high-resolution image
    upscaled_image = upscaler_pipeline(prompt=prompt, image=low_res_latents, num_inference_steps=20, guidance_scale=0).images[0]

    return upscaled_image

# test_function_code --------------------

def test_upscale_movie_character_images():
    print("Testing started.")

    # Test case 1: Upscaling a low-resolution image of a movie character
    print("Testing case [1/1] started.")
    prompt = "a photo of The Joker, high-resolution, detailed"
    upscaled_image = upscale_movie_character_images(prompt)
    assert upscaled_image is not None, f"Test case [1/1] failed: Expected upscaled image, got None."
    print("Testing finished.")

# call_test_function_line --------------------

test_upscale_movie_character_images()