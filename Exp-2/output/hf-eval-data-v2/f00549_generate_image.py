# function_import --------------------

from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def generate_image(prompt: str, model_name: str = 'runwayml/stable-diffusion-inpainting', revision: str = 'fp16', torch_dtype = torch.float16) -> Image:
    """
    Generate an image based on a text prompt using a pre-trained model.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_name (str, optional): The name of the pre-trained model to use. Defaults to 'runwayml/stable-diffusion-inpainting'.
        revision (str, optional): The revision of the model to use. Defaults to 'fp16'.
        torch_dtype (torch.dtype, optional): The data type to use in torch. Defaults to torch.float16.

    Returns:
        Image: The generated image.
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, revision=revision, torch_dtype=torch_dtype)
    image = pipe(prompt=prompt).images[0]
    return image

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.
    """
    prompt = 'kangaroo eating pizza'
    image = generate_image(prompt)
    assert isinstance(image, Image.Image)

# call_test_function_code --------------------

test_generate_image()