# function_import --------------------

from diffusers import StableDiffusionInpaintPipeline
import torch
import os

# function_code --------------------

def generate_image(prompt: str, model_name: str = 'runwayml/stable-diffusion-inpainting', revision: str = 'fp16', torch_dtype = torch.float16) -> None:
    '''
    Generate an image based on the given text prompt using the StableDiffusionInpaintPipeline.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_name (str, optional): The name of the pre-trained model to use. Defaults to 'runwayml/stable-diffusion-inpainting'.
        revision (str, optional): The revision of the model to use. Defaults to 'fp16'.
        torch_dtype (torch.dtype, optional): The data type to use in torch. Defaults to torch.float16.

    Returns:
        None. The function saves the generated image to a file.
    '''
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, revision=revision, torch_dtype=torch_dtype)
    image = pipe(prompt=prompt).images[0]
    image.save(f'{prompt.replace(' ', '_')}_sign.png')

# test_function_code --------------------

def test_generate_image():
    '''
    Test the generate_image function.
    '''
    generate_image('kangaroo eating pizza')
    assert os.path.exists('kangaroo_eating_pizza_sign.png')
    generate_image('cat playing guitar')
    assert os.path.exists('cat_playing_guitar_sign.png')
    generate_image('dog riding skateboard')
    assert os.path.exists('dog_riding_skateboard_sign.png')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image()