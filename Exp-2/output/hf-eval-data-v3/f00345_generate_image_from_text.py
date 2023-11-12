# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch
import os

# function_code --------------------

def generate_image_from_text(prompt: str, model_id: str = 'prompthero/openjourney', save_path: str = './generated_image.png'):
    """
    Generate an image from a text prompt using a pre-trained model.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str, optional): The id of the pre-trained model to use. Defaults to 'prompthero/openjourney'.
        save_path (str, optional): The path to save the generated image. Defaults to './generated_image.png'.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the required libraries are not installed.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    image = pipe(prompt).images[0]
    image.save(save_path)

# test_function_code --------------------

def test_generate_image_from_text():
    """
    Test the function generate_image_from_text.

    Returns:
        str: 'All Tests Passed' if all assertions pass, otherwise the assertion error is raised.
    """
    # Test with default parameters
    generate_image_from_text('A vintage sports car racing through a desert landscape during sunset')
    assert os.path.exists('./generated_image.png')
    os.remove('./generated_image.png')

    # Test with custom parameters
    generate_image_from_text('A vintage sports car racing through a desert landscape during sunset', 'prompthero/openjourney', './custom_path.png')
    assert os.path.exists('./custom_path.png')
    os.remove('./custom_path.png')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image_from_text()