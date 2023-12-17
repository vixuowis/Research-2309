# requirements_file --------------------

!pip install -U torch,diffusers

# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_astronaut_playing_guitar_image(model_id: str, prompt: str) -> 'Image':
    """
    Generates an image based on the given text prompt using the specified model.

    Parameters:
    model_id (str): The identifier for the pre-trained model.
    prompt (str): The text prompt to generate the image from.

    Returns:
    Image: The generated image.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    generated_image = pipe(prompt).images[0]
    return generated_image

# test_function_code --------------------

def test_generate_astronaut_playing_guitar_image():
    print("Testing generate_astronaut_playing_guitar_image function.")
    model_id = 'dreamlike-art/dreamlike-photoreal-2.0'
    prompt = 'astronaut playing guitar in space'
    image = generate_astronaut_playing_guitar_image(model_id, prompt)

    # Test case 1: The function should return an Image object
    print("Test case 1: Checking the returned type.")
    assert isinstance(image, Image), "Test case 1 failed: The function did not return an Image object."

    # Additional test cases can be added as needed
    print("All test cases passed.")

# Run the test function
test_generate_astronaut_playing_guitar_image()