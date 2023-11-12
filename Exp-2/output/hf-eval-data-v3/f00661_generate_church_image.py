# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_church_image(model_id: str = 'google/ddpm-ema-church-256') -> None:
    """
    Generate an image of a church using a pretrained DDPM model.

    Args:
        model_id (str): The ID of the pretrained model to use. Default is 'google/ddpm-ema-church-256'.

    Returns:
        None. The function saves the generated image to a file.

    Raises:
        ModuleNotFoundError: If the diffusers library is not installed.
    """
    ddpm = DDPMPipeline.from_pretrained(model_id)
    generated_image_result = ddpm()
    image = generated_image_result.images[0]
    image.save('ddpm_generated_church_image.png')

# test_function_code --------------------

def test_generate_church_image():
    """
    Test the generate_church_image function.

    Returns:
        str: 'All Tests Passed' if all assertions pass.
    """
    import os

    # Test with default model_id
    generate_church_image()
    assert os.path.exists('ddpm_generated_church_image.png'), 'Image not generated with default model_id'

    # Test with a different model_id
    generate_church_image('google/ddpm-ema-church-512')
    assert os.path.exists('ddpm_generated_church_image.png'), 'Image not generated with different model_id'

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_church_image())