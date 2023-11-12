# function_import --------------------

import numpy as np
from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_image():
    """
    This function generates an image of a cute butterfly using the pre-trained model 'clp/sd-class-butterflies-32'.
    
    Returns:
        numpy.ndarray: The generated butterfly image.
    """
    pipeline = DDPMPipeline.from_pretrained('clp/sd-class-butterflies-32')
    image = pipeline().images[0]
    return image

# test_function_code --------------------

def test_generate_butterfly_image():
    """
    This function tests the 'generate_butterfly_image' function by checking the type of the returned image.
    """
    image = generate_butterfly_image()
    assert isinstance(image, np.ndarray), 'The returned object is not a numpy array.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_generate_butterfly_image()