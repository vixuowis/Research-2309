# requirements_file --------------------

!pip install -U diffusers transformers scipy

# function_import --------------------

import torch
from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_futuristic_ocean_city_image(prompt: str, model_id: str = 'CompVis/stable-diffusion-v1-4', device: str = 'cuda') -> 'Image':
    """
    Generate an image of a futuristic city under the ocean using a text-to-image model.
    
    Args:
        prompt (str): A text prompt describing the image to be generated.
        model_id (str, optional): The model identifier for the text-to-image model. Defaults to 'CompVis/stable-diffusion-v1-4'.
        device (str, optional): The computing device to use for image generation. Defaults to 'cuda'.
    
    Returns:
        Image: The generated image of a futuristic city under the ocean.
    
    Raises:
        RuntimeError: If there is an error loading the model or generating the image.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]
    return image

# test_function_code --------------------

def test_generate_futuristic_ocean_city_image():
    print("Testing started.")

    prompt = 'A futuristic city under the ocean'  # The prompt for the test
    expected_output_type = 'PIL.JpegImagePlugin.JpegImageFile'  # The expected image type

    # Test case 1: Test if the function generates an image
    print("Testing case [1/1] started.")
    result = generate_futuristic_ocean_city_image(prompt)
    assert isinstance(result, expected_output_type), f"Test case [1/1] failed: Expected image type not generated"

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_futuristic_ocean_city_image()