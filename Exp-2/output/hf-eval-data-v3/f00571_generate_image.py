# function_import --------------------

import os
from diffusers import DiffusionPipeline

# function_code --------------------

def generate_image(model_id: str, num_inference_steps: int, save_path: str):
    """
    Generate high-quality images of faces using a pre-trained model.

    Args:
        model_id (str): The id of the pre-trained model.
        num_inference_steps (int): The number of inference steps.
        save_path (str): The path to save the generated image.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the 'diffusers' module is not installed.
    """
    pipeline = DiffusionPipeline.from_pretrained(model_id)
    image = pipeline(num_inference_steps=num_inference_steps)
    image[0].save(save_path)

# test_function_code --------------------

def test_generate_image():
    """
    Test the 'generate_image' function.

    Returns:
        str: 'All Tests Passed' if all assertions pass, otherwise the assertion error message.
    """
    try:
        generate_image('CompVis/ldm-celebahq-256', 200, 'test_image.png')
        assert os.path.exists('test_image.png')
        os.remove('test_image.png')
    except AssertionError as e:
        return str(e)
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_image())