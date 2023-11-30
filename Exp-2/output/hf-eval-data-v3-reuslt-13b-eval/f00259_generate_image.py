# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline
import os

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

    print('Loading Model...')
    
    # Load model and tokenizer from HuggingFace Hub
    pipeline = StableDiffusionPipeline(model_id, device=device)
    
    try:
        os.makedirs('images/generated-images')
    except OSError as error:
        print()
        
    # Generate the image
    model_output = pipeline.generate(prompt)

    # Save the generated image to the current directory
    model_output['image'].save("./images/generated-images/generated-image.png")
    

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.
    """
    generate_image('A futuristic city under the ocean')
    assert os.path.exists('A_futuristic_city_under_the_ocean.png')
    os.remove('A_futuristic_city_under_the_ocean.png')
    generate_image('An astronaut riding a horse on mars')
    assert os.path.exists('An_astronaut_riding_a_horse_on_mars.png')
    os.remove('An_astronaut_riding_a_horse_on_mars.png')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_image()