# function_import --------------------

from diffusers import DDPMPipeline
import PIL.Image

# function_code --------------------

def generate_butterfly_image():
    """
    Generate images of cute butterflies using the 'myunus1/diffmodels_galaxies_scratchbook' model.

    Returns:
        PIL.Image.Image: The generated image of a butterfly.
    """
    pipeline = DDPMPipeline.from_pretrained('myunus1/diffmodels_galaxies_scratchbook')
    generated_data = pipeline()
    image = generated_data.images[0]
    return image

# test_function_code --------------------

def test_generate_butterfly_image():
    """
    Test the 'generate_butterfly_image' function.
    """
    image = generate_butterfly_image()
    assert isinstance(image, PIL.Image.Image), 'The function should return a PIL image.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_butterfly_image()