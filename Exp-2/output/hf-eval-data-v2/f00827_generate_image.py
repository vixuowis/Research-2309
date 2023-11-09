# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'CompVis/stable-diffusion-v1-4', device: str = 'cuda') -> None:
    """
    Generate an image based on the given text prompt using the StableDiffusionPipeline.

    Args:
        prompt (str): The text description of the image to be generated.
        model_id (str, optional): The model to be used for image generation. Defaults to 'CompVis/stable-diffusion-v1-4'.
        device (str, optional): The device to be used for image generation. Defaults to 'cuda'.

    Returns:
        None. The function saves the generated image to a file.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]
    image.save('generated_image.png')

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.

    The function does not return any value. It saves the generated image to a file. The test will pass if the function runs without raising an exception.
    """
    try:
        generate_image('a futuristic 3D printed car')
        print('Test passed.')
    except Exception as e:
        print(f'Test failed. {str(e)}')

# call_test_function_code --------------------

test_generate_image()