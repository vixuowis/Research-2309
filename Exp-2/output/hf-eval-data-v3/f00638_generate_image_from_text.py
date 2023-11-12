# function_import --------------------

from diffusers import StableDiffusionInpaintPipeline
import torch
import os

# function_code --------------------

def generate_image_from_text(prompt: str, image_path: str = None, mask_image_path: str = None) -> None:
    """
    Generate an image based on a text description using the StableDiffusionInpaintPipeline from Hugging Face.

    Args:
        prompt (str): The text description to generate the image from.
        image_path (str, optional): The path to the initial image. Defaults to None.
        mask_image_path (str, optional): The path to the mask image. Defaults to None.

    Returns:
        None. The generated image is saved as './generated_image.png'.
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16)
    pipe.to('cuda')
    image, mask_image = None, None
    if image_path:
        image = torch.load(image_path)
    if mask_image_path:
        mask_image = torch.load(mask_image_path)
    output_image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    output_image.save('./generated_image.png')

# test_function_code --------------------

def test_generate_image_from_text():
    """
    Test the function generate_image_from_text.
    """
    generate_image_from_text('A beautiful landscape with a waterfall and a sunset')
    assert os.path.exists('./generated_image.png'), 'Image not generated.'
    os.remove('./generated_image.png')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image_from_text()