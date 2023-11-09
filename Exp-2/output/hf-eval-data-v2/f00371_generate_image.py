# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_image(model_id: str) -> None:
    """
    Generate a high-quality image of a church using unconditional image generation.

    Args:
        model_id (str): The model id of the pretrained model.

    Returns:
        None. The function saves the generated image to a file named 'ddpm_generated_image.png'.
    """
    ddpm = DDPMPipeline.from_pretrained(model_id)
    image = ddpm().images[0]
    image.save('ddpm_generated_image.png')

# test_function_code --------------------

def test_generate_image():
    """
    Test the function generate_image.

    The function will raise an error if the image is not successfully generated and saved.
    """
    model_id = 'google/ddpm-church-256'
    generate_image(model_id)
    assert os.path.exists('ddpm_generated_image.png'), 'Image not generated.'

# call_test_function_code --------------------

test_generate_image()