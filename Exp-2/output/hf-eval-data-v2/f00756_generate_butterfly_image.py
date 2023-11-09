# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

butterfly_generator = DDPMPipeline.from_pretrained('ocariz/butterfly_200')

def generate_butterfly_image():
    """
    This function generates an image of a butterfly using the 'ocariz/butterfly_200' model from Hugging Face Transformers.
    The model is a diffusion model for unconditional image generation of butterflies trained for 200 epochs.
    
    Returns:
        numpy.ndarray: The generated butterfly image.
    """
    butterfly_image = butterfly_generator().images[0]
    return butterfly_image

# test_function_code --------------------

def test_generate_butterfly_image():
    """
    This function tests the 'generate_butterfly_image' function by generating a butterfly image and checking if the output is a numpy.ndarray.
    """
    butterfly_image = generate_butterfly_image()
    assert isinstance(butterfly_image, np.ndarray), 'The output should be a numpy.ndarray.'

# call_test_function_code --------------------

test_generate_butterfly_image()