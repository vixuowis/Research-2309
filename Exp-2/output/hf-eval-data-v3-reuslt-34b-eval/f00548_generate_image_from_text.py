# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline
import os

# function_code --------------------

def generate_image_from_text(prompt: str, model_id: str = 'CompVis/stable-diffusion-v1-4', save_path: str = 'generated_image.png'):
    """
    Generate an image from a text description using the StableDiffusionPipeline from Hugging Face.

    Args:
        prompt (str): The text description of the image to generate.
        model_id (str, optional): The model id of the pretrained model to use. Defaults to 'CompVis/stable-diffusion-v1-4'.
        save_path (str, optional): The path to save the generated image. Defaults to 'generated_image.png'.

    Returns:
        None
    """
    
    # create pipeline object
    pipe = StableDiffusionPipeline(model=model_id)
    
    # create random tensor to pass through model, use betas to control noise levels
    torch.manual_seed(0)
    noise = torch.randn([1, 3, 256, 256])
    
    # generate image from text and store it in a directory
    image = pipe(noise=noise, prompts=[prompt], batch_size=1)[0]
    os.makedirs('generated', exist_ok=True)
    torchvision.utils.save_image(image[0], 'generated/'+save_path)

# test_function_code --------------------

def test_generate_image_from_text():
    """
    Test the function generate_image_from_text.
    """
    generate_image_from_text('a serene lake at sunset', save_path='serene_lake_sunset.png')
    assert os.path.exists('serene_lake_sunset.png'), 'Image not generated!'
    os.remove('serene_lake_sunset.png')
    generate_image_from_text('an astronaut riding a horse on mars', save_path='astronaut_rides_horse.png')
    assert os.path.exists('astronaut_rides_horse.png'), 'Image not generated!'
    os.remove('astronaut_rides_horse.png')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_image_from_text()