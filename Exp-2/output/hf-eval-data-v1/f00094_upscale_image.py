from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
import torch

def upscale_image(prompt):
    '''
    This function takes a text prompt as input and returns an upscaled high-resolution image.
    It uses the StableDiffusionLatentUpscalePipeline from Hugging Face to upscale low-resolution images.
    '''
    # Create a pipeline using the pretrained 'CompVis/stable-diffusion-v1-4' model to generate the low-resolution latent image
    pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
    pipeline.to('cuda')
    
    # Create an instance of the StableDiffusionLatentUpscalePipeline with the pretrained 'stabilityai/sd-x2-latent-upscaler' model
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained('stabilityai/sd-x2-latent-upscaler', torch_dtype=torch.float16)
    upscaler.to('cuda')
    
    # Generate the low-resolution latent image
    generator = torch.manual_seed(33)
    low_res_latents = pipeline(prompt, generator=generator, output_type='latent').images
    
    # Generate the upscaled high-resolution image
    upscaled_image = upscaler(prompt=prompt, image=low_res_latents, num_inference_steps=20, guidance_scale=0, generator=generator).images[0]
    
    # Save the upscaled high-resolution image as a .png file
    upscaled_image.save('movie_character_high_resolution.png')
    return upscaled_image