# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_image():
    """
    Generate images of cute butterflies using the myunus1/diffmodels_galaxies_scratchbook model.

    Returns:
        PIL.Image: An image of a generated butterfly.
    """
    pipeline = DDPMPipeline.from_pretrained('myunus1/diffmodels_galaxies_scratchbook')
    generated_data = pipeline()
    image = generated_data.images[0]
    return image

# test_function_code --------------------

def test_generate_butterfly_image():
    """
    Test the generate_butterfly_image function.

    Raises:
        AssertionError: If the function does not return an instance of PIL.Image.
    """
    image = generate_butterfly_image()
    assert isinstance(image, PIL.Image), 'The function should return a PIL.Image instance.'

# call_test_function_code --------------------

test_generate_butterfly_image()