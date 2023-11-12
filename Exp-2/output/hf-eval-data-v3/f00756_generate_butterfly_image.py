# function_import --------------------

import numpy as np
from diffusers import DDPMPipeline

# function_code --------------------

butterfly_generator = DDPMPipeline.from_pretrained('ocariz/butterfly_200')

def generate_butterfly_image():
    """
    Generate an image of a butterfly.

    This function uses the 'ocariz/butterfly_200' model from Hugging Face Transformers to generate an image of a butterfly.

    Returns:
        numpy.ndarray: The generated butterfly image.
    """
    butterfly_image = butterfly_generator().images[0]
    return butterfly_image

# test_function_code --------------------

def test_generate_butterfly_image():
    """
    Test the 'generate_butterfly_image' function.

    This function tests the 'generate_butterfly_image' function by calling it and checking the type of the returned value.
    """
    butterfly_image = generate_butterfly_image()
    assert isinstance(butterfly_image, np.ndarray), 'The returned value is not a numpy array.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_butterfly_image()