# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline
import os

# function_code --------------------

def generate_image_from_text(prompt: str, model_id: str = 'CompVis/stable-diffusion-v1-4', save_path: str = 'generated_image.png'):
    """
    Generate an image from a text description using the StableDiffusionPipeline from Hugging Face.

    Args:
        prompt (str): The text description of the image to generate.
        model_id (str, optional): The model id of the pretrained model to use. Defaults to 'CompVis/stable-diffusion-v1-4'.
        save_path (str, optional): The path to save the generated image. Defaults to 'generated_image.png'.

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]
    image.save(save_path)

# test_function_code --------------------

def test_generate_image_from_text():
    """
    Test the function generate_image_from_text.
    """
    generate_image_from_text('a serene lake at sunset', save_path='serene_lake_sunset.png')
    assert os.path.exists('serene_lake_sunset.png'), 'Image not generated!'
    os.remove('serene_lake_sunset.png')
    generate_image_from_text('an astronaut riding a horse on mars', save_path='astronaut_rides_horse.png')
    assert os.path.exists('astronaut_rides_horse.png'), 'Image not generated!'
    os.remove('astronaut_rides_horse.png')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image_from_text()