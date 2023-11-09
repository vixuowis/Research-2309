# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'CompVis/stable-diffusion-v1-4', device: str = 'cuda') -> None:
    """
    Generate an image based on a text prompt using the StableDiffusionPipeline from Hugging Face.

    Args:
        prompt (str): The text description of the image to generate.
        model_id (str, optional): The id of the pretrained model to use. Defaults to 'CompVis/stable-diffusion-v1-4'.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        None. The function saves the generated image to the current directory.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]
    image.save(f'{prompt.replace(" ", "_")}.png')

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.

    The function generates an image based on a text prompt and saves it to the current directory.
    The test checks if the image file is created.
    """
    import os
    prompt = 'a serene lake at sunset'
    generate_image(prompt)
    assert os.path.isfile(f'{prompt.replace(" ", "_")}.png')

# call_test_function_code --------------------

test_generate_image()