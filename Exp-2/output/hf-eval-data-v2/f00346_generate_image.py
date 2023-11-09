# function_import --------------------

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'stabilityai/stable-diffusion-2-1') -> Image:
    """
    Generate an image based on a text description using a pre-trained model.

    Args:
        prompt (str): Text description of the scene.
        model_id (str, optional): ID of the pre-trained model. Defaults to 'stabilityai/stable-diffusion-2-1'.

    Returns:
        Image: Generated image.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cuda')
    generated_image = pipe(prompt).images[0]
    return generated_image

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.
    """
    prompt = 'a scene of a magical forest with fairies and elves'
    generated_image = generate_image(prompt)
    assert isinstance(generated_image, Image.Image), 'The output should be an instance of PIL.Image.Image.'

# call_test_function_code --------------------

test_generate_image()