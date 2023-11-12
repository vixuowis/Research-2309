# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch
import os

# function_code --------------------

def generate_anime_style_image(prompt: str, model_id: str = 'andite/anything-v4.0', device: str = 'cuda') -> None:
    """
    Generate an anime-style image based on the given prompt and save it to a file.

    Args:
        prompt (str): The text prompt to be used for image generation.
        model_id (str, optional): The model id to be used for image generation. Defaults to 'andite/anything-v4.0'.
        device (str, optional): The device to be used for image generation. Defaults to 'cuda'.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the required modules are not found.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    generated_image = pipe(prompt).images[0]
    generated_image.save('./anime_girl_guitar.png')

# test_function_code --------------------

def test_generate_anime_style_image():
    """
    Test the function generate_anime_style_image.

    Returns:
        str: 'All Tests Passed' if all assertions pass, otherwise the error message.
    """
    try:
        generate_anime_style_image('anime-style girl with a guitar')
        assert os.path.exists('./anime_girl_guitar.png')
        os.remove('./anime_girl_guitar.png')
        return 'All Tests Passed'
    except Exception as e:
        return str(e)

# call_test_function_code --------------------

print(test_generate_anime_style_image())