# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline
import os

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'CompVis/stable-diffusion-v1-4', device: str = 'cuda', save_path: str = 'generated_image.png'):
    '''
    Generate an image based on the provided text prompt using the StableDiffusionPipeline.

    Args:
        prompt (str): The text description of the image to be generated.
        model_id (str, optional): The model to be used for image generation. Defaults to 'CompVis/stable-diffusion-v1-4'.
        device (str, optional): The device to be used for image generation. Defaults to 'cuda'.
        save_path (str, optional): The path where the generated image will be saved. Defaults to 'generated_image.png'.

    Returns:
        None
    '''
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]
    image.save(save_path)

# test_function_code --------------------

def test_generate_image():
    '''
    Test the generate_image function.
    '''
    generate_image('a futuristic 3D printed car', save_path='3D_printed_car.png')
    assert os.path.exists('3D_printed_car.png'), 'Test Failed: Image not generated!'
    os.remove('3D_printed_car.png')
    generate_image('an astronaut riding a horse on mars', save_path='astronaut_rides_horse.png')
    assert os.path.exists('astronaut_rides_horse.png'), 'Test Failed: Image not generated!'
    os.remove('astronaut_rides_horse.png')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image()