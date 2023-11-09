# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_anime_style_image(prompt: str, model_id: str = 'andite/anything-v4.0', device: str = 'cuda') -> None:
    """
    Generate a character-like artwork image based on the given prompt using the StableDiffusionPipeline.

    Args:
        prompt (str): The text prompt to be used for image generation.
        model_id (str, optional): The model id to be used for image generation. Defaults to 'andite/anything-v4.0'.
        device (str, optional): The device to be used for image generation. Defaults to 'cuda'.

    Returns:
        None. The function saves the generated image to a file.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    generated_image = pipe(prompt).images[0]
    generated_image.save('./anime_girl_guitar.png')

# test_function_code --------------------

def test_generate_anime_style_image():
    """
    Test the function generate_anime_style_image.

    The function is tested with a known prompt 'anime-style girl with a guitar'.
    The test passes if the function executes without throwing an exception.
    """
    try:
        generate_anime_style_image('anime-style girl with a guitar')
        print('Test passed.')
    except Exception as e:
        print('Test failed. The function threw an exception:')
        print(e)

# call_test_function_code --------------------

test_generate_anime_style_image()