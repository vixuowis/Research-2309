# requirements_file --------------------

!pip install -U diffusers transformers scipy

# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_futuristic_city_image(prompt):
    '''
    Generates an image of a futuristic city under the ocean based on a text prompt using Stable Diffusion model.

    Args:
    prompt (str): A text prompt describing the futuristic city.

    Returns:
    Image: The generated image of the futuristic city.
    '''
    model_id = 'CompVis/stable-diffusion-v1-4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]
    return image

# test_function_code --------------------

def test_generate_futuristic_city_image():
    print("Testing generate_futuristic_city_image function.")
    prompt = 'A futuristic city under the ocean'
    image = generate_futuristic_city_image(prompt)
    assert image is not None, "The generated image is None."
    assert isinstance(image, Image.Image), "The generated object is not an image."
    print("Test passed successfully.")