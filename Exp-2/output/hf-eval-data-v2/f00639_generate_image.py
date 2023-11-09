# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'dreamlike-art/dreamlike-photoreal-2.0') -> None:
    """
    Generate an image based on a textual prompt using a pre-trained model.

    Args:
        prompt (str): The textual description of the desired image.
        model_id (str, optional): The ID of the pre-trained model to use. Defaults to 'dreamlike-art/dreamlike-photoreal-2.0'.

    Returns:
        None. The function saves the generated image to the current directory.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    generated_image = pipe(prompt).images[0]
    generated_image.save('result.png')

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.

    The function does not return a value, so the test will pass if the function
    runs without raising an exception.
    """
    test_prompt = 'astronaut playing guitar in space'
    generate_image(test_prompt)

# call_test_function_code --------------------

test_generate_image()