# function_import --------------------

from diffusers import DiffusionPipeline

# function_code --------------------

def generate_image(model_id: str, num_inference_steps: int = 200) -> None:
    """
    Generate a high-quality image of a face using a pre-trained model.

    Args:
        model_id (str): The ID of the pre-trained model to use for image generation.
        num_inference_steps (int, optional): The number of inference steps to use. Defaults to 200.

    Returns:
        None. The function saves the generated image as 'ldm_generated_image.png'.
    """
    pipeline = DiffusionPipeline.from_pretrained(model_id)
    image = pipeline(num_inference_steps=num_inference_steps)
    image[0].save('ldm_generated_image.png')

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.

    The function does not return a value, so the test will pass if the function
    runs without raising an exception.
    """
    model_id = 'CompVis/ldm-celebahq-256'
    try:
        generate_image(model_id)
    except Exception as e:
        assert False, f'Exception occurred: {e}'

# call_test_function_code --------------------

test_generate_image()