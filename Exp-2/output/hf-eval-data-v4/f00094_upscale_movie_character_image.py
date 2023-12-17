# requirements_file --------------------

!pip install -U diffusers torch

# function_import --------------------

from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch

# function_code --------------------

def upscale_movie_character_image(prompt, seed=33):
    # Load the pipeline for initial low-resolution image generation
    pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
    pipeline.to('cuda')
    # Load the upscaling pipeline
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained('stabilityai/sd-x2-latent-upscaler', torch_dtype=torch.float16)
    upscaler.to('cuda')
    # Set a seed for reproducibility
    generator = torch.manual_seed(seed)
    # Generate the low-resolution latent image
    low_res_latents = pipeline(prompt, generator=generator, output_type='latent').images
    # Generate the upscaled high-resolution image
    upscaled_image = upscaler(prompt=prompt, image=low_res_latents, num_inference_steps=20, guidance_scale=0, generator=generator).images[0]
    # Save the high-resolution image
    upscaled_image.save(f'{prompt}_high_resolution.png')
    return upscaled_image

# test_function_code --------------------

def test_upscale_movie_character_image():
    print('Testing upscale_movie_character_image function.')
    # Example prompt to test with
    test_prompt = 'a photo of a movie character'
    # Function call with test data
    output_image = upscale_movie_character_image(test_prompt)
    # We cannot directly assert the correctness of the image, so we check if an image object is returned
    assert output_image is not None, 'Failed to create an upscaled image.'
    print('Test passed. An upscaled image was created successfully.')

# Run the test
if __name__ == '__main__':
    test_upscale_movie_character_image()