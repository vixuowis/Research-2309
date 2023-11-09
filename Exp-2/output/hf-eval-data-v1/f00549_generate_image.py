from diffusers import StableDiffusionInpaintPipeline
import torch


def generate_image(prompt):
    '''
    This function generates an image based on the given text prompt using the StableDiffusionInpaintPipeline from Hugging Face.
    
    Parameters:
    prompt (str): The text prompt to generate the image from.
    
    Returns:
    PIL image: The generated image.
    '''
    # Load the pre-trained model into the pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting', revision='fp16', torch_dtype=torch.float16)
    
    # Generate the image
    image = pipe(prompt=prompt).images[0]
    
    # Save the image
    image.save('kangaroo_pizza_sign.png')
    
    return image