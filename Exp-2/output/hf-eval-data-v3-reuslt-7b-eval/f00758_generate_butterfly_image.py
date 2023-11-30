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
    diffuser = DDPMPipeline('myunus1', '/home/gst-ai/Documents/diffusers/models/myunus1/diffmodels_galaxies_scratchbook')
    
    # generate and return an image of a butterfly (256 x 256 px)
    return diffuser.generate(256, 256, 16)

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