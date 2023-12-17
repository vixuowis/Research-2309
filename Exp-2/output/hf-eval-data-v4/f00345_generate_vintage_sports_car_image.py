# requirements_file --------------------

!pip install -U diffusers, torch

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_vintage_sports_car_image(prompt):
    """
    Generates an image of a vintage sports car racing through a desert landscape during sunset.

    Parameters:
    prompt (str): Description of the image to generate.

    Returns:
    Image: An image object of the generated scene.
    """
    model_id = 'prompthero/openjourney'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    image = pipe(prompt).images[0]
    return image

# test_function_code --------------------

def test_generate_vintage_sports_car_image():
    print("Testing generate_vintage_sports_car_image function.")

    # Test case: Generate image with specified prompt
    prompt = 'A vintage sports car racing through a desert landscape during sunset'
    print("Test case started.")
    image = generate_vintage_sports_car_image(prompt)
    assert image is not None, "Test case failed: Image is None"
    image.save('./vintage_sports_car_desert_sunset.png')
    print("Test case passed. Image saved as 'vintage_sports_car_desert_sunset.png'")

    print("Testing finished.")

# Run the test function
test_generate_vintage_sports_car_image()