# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_insect_image(model_name: str = 'schdoel/sd-class-AFHQ-32') -> None:
    """
    Generate an insect image using a pretrained model.

    Args:
        model_name (str): The name of the pretrained model. Default is 'schdoel/sd-class-AFHQ-32'.

    Returns:
        None. The function saves the generated image to the current directory.
    """
    pipeline = DDPMPipeline.from_pretrained(model_name)
    generated_image = pipeline().images[0]
    generated_image.save('insect_image.png')

# test_function_code --------------------

def test_generate_insect_image():
    """
    Test the generate_insect_image function.

    The function does not return a value, so the test will pass if the function runs without errors.
    """
    generate_insect_image()

# call_test_function_code --------------------

test_generate_insect_image()