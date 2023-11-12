# function_import --------------------

import os
from diffusers import StableDiffusionInpaintPipeline
import torch

# function_code --------------------

def generate_image_from_text(prompt: str, save_path: str = 'generated_image.png'):
    '''
    Generate an image based on the given text prompt using the StableDiffusionInpaintPipeline from the diffusers package.

    Args:
        prompt (str): The text description of the desired image.
        save_path (str): The path where the generated image will be saved. Default is 'generated_image.png'.

    Returns:
        None. The generated image is saved to the specified path.
    '''
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16)
    pipe.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    generated_image = pipe(prompt=prompt).images[0]
    generated_image.save(save_path)

# test_function_code --------------------

def test_generate_image_from_text():
    '''
    Test the generate_image_from_text function.
    '''
    # Test case 1: A modern living room with a fireplace and a large window overlooking a forest.
    generate_image_from_text('A modern living room with a fireplace and a large window overlooking a forest.', 'test_image_1.png')
    assert os.path.exists('test_image_1.png')

    # Test case 2: A serene beach with clear blue water and white sand.
    generate_image_from_text('A serene beach with clear blue water and white sand.', 'test_image_2.png')
    assert os.path.exists('test_image_2.png')

    # Test case 3: A bustling city street at night with bright neon lights.
    generate_image_from_text('A bustling city street at night with bright neon lights.', 'test_image_3.png')
    assert os.path.exists('test_image_3.png')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image_from_text()