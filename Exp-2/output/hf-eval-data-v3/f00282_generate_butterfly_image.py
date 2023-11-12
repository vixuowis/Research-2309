# function_import --------------------

import numpy as np
from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_image():
    """
    This function generates an image of a butterfly using a pre-trained model from Hugging Face Transformers.

    Returns:
        numpy.ndarray: The generated image of the butterfly.
    """
    pipeline = DDPMPipeline.from_pretrained('utyug1/sd-class-butterflies-32')
    generated_image = pipeline().images[0]
    return generated_image

# test_function_code --------------------

def test_generate_butterfly_image():
    """
    This function tests the generate_butterfly_image function by checking if the output is a numpy array.
    """
    generated_image = generate_butterfly_image()
    assert isinstance(generated_image, np.ndarray), 'The output should be a numpy array.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_generate_butterfly_image()