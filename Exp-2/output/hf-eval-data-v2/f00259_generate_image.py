# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'CompVis/stable-diffusion-v1-4', device: str = 'cuda') -> None:
    """
    Generate an image based on the given text prompt using the StableDiffusionPipeline model.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str, optional): The model id to use for the image generation. Defaults to 'CompVis/stable-diffusion-v1-4'.
        device (str, optional): The device to run the model on. Defaults to 'cuda'.

    Returns:
        None. The function saves the generated image to the current directory.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]
    image.save('generated_image.png')

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.

    The function does not return anything. The test will pass if the function runs without raising an exception.
    """
    try:
        generate_image('A futuristic city under the ocean')
    except Exception as e:
        assert False, f'Exception occurred: {e}'

# call_test_function_code --------------------

test_generate_image()