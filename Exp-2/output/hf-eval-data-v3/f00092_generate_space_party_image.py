# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch
import os

# function_code --------------------

def generate_space_party_image(prompt: str, save_path: str = 'space_party.png'):
    '''
    Generate an image based on the given text prompt using the StableDiffusionPipeline model.

    Args:
        prompt (str): The text prompt to generate the image from.
        save_path (str): The path to save the generated image. Default is 'space_party.png'.

    Returns:
        None. The function saves the generated image to the specified path.

    Raises:
        ModuleNotFoundError: If the required 'diffusers' module is not found.
    '''
    pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1', torch_dtype=torch.float16)
    image = pipe(prompt).images[0]
    image.save(save_path)

# test_function_code --------------------

def test_generate_space_party_image():
    '''
    Test the generate_space_party_image function.
    '''
    generate_space_party_image('a space party with astronauts and aliens having fun together', 'test_space_party.png')
    assert os.path.exists('test_space_party.png'), 'Test image not found.'
    os.remove('test_space_party.png')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_space_party_image()