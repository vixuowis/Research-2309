# function_import --------------------

from diffusers import StableDiffusionPipeline

# function_code --------------------

def generate_space_party_image(prompt: str, model_id: str = 'stabilityai/stable-diffusion-2-1', dtype: str = 'torch.float16') -> None:
    """
    Generate an image based on a text prompt using a pre-trained model from Hugging Face.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str, optional): The identifier of the pre-trained model. Defaults to 'stabilityai/stable-diffusion-2-1'.
        dtype (str, optional): The data type for the torch tensor. Defaults to 'torch.float16'.

    Returns:
        None. The function saves the generated image locally.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    image = pipe(prompt).images[0]
    image.save('space_party.png')

# test_function_code --------------------

def test_generate_space_party_image():
    """
    Test the generate_space_party_image function.

    The function does not return any value. The test will pass if the function runs without raising an error.
    """
    prompt = 'a space party with astronauts and aliens having fun together'
    try:
        generate_space_party_image(prompt)
    except Exception as e:
        assert False, f'Exception occurred: {e}'

# call_test_function_code --------------------

test_generate_space_party_image()