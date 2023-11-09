# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_image(model_id: str = 'clp/sd-class-butterflies-32') -> None:
    """
    Generate a butterfly image using a pre-trained model from Hugging Face Transformers.

    Args:
        model_id: The identifier of the pre-trained model. Default is 'clp/sd-class-butterflies-32'.

    Returns:
        None. The function saves the generated image to the current directory.
    """
    pipeline = DDPMPipeline.from_pretrained(model_id)
    image = pipeline().images[0]
    image.save('cute_butterfly_image.png')

# test_function_code --------------------

def test_generate_butterfly_image():
    """
    Test the generate_butterfly_image function.

    The function should not raise any exceptions.
    """
    try:
        generate_butterfly_image()
    except Exception as e:
        assert False, f'Exception occurred: {e}'

# call_test_function_code --------------------

test_generate_butterfly_image()