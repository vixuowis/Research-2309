from diffusers import StableDiffusionInpaintPipeline
import torch

def generate_image_from_text(prompt: str, image_path: str = None, mask_image_path: str = None) -> None:
    '''
    This function generates an image based on a text description using the StableDiffusionInpaintPipeline from Hugging Face.
    
    Args:
    prompt (str): The text description based on which the image will be generated.
    image_path (str, optional): The path to the image file if any. Defaults to None.
    mask_image_path (str, optional): The path to the mask image file if any. Defaults to None.
    
    Returns:
    None
    '''
    # Load the pretrained stable-diffusion-2-inpainting model
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16)
    pipe.to('cuda')
    
    # If image and mask image paths are provided, load the images
    image, mask_image = None, None
    if image_path:
        image = torch.load(image_path)
    if mask_image_path:
        mask_image = torch.load(mask_image_path)
    
    # Generate the image based on the text prompt
    output_image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    
    # Save the generated image
    output_image.save('./generated_image.png')