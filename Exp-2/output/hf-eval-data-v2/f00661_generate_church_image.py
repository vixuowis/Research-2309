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
    """
    ddpm = DDPMPipeline.from_pretrained(model_id)
    generated_image_result = ddpm()
    image = generated_image_result.images[0]
    image.save('ddpm_generated_church_image.png')

# test_function_code --------------------

def test_generate_church_image():
    """
    Test the generate_church_image function.

    The function should not raise any exceptions.
    """
    try:
        generate_church_image()
    except Exception as e:
        assert False, f'Exception occurred: {e}'

# call_test_function_code --------------------

test_generate_church_image()