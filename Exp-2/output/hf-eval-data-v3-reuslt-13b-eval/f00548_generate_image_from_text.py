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

    # Load pretrained diffusion model from Hugging Face Hub
    diffusion = StableDiffusionPipeline(model=model_id)

    # Generate the image with a given prompt
    images = diffusion.sample(description=prompt, batch_size=1)
    
    for image in images:
        if os.path.exists(save_path):
            print("Overwriting existing file.")
            
        torchvision.utils.save_image(torch.tensor([image]), save_path)

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