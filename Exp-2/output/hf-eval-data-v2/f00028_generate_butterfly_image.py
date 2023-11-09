# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_image():
    """
    This function generates an image of a cute butterfly using the pre-trained model 'clp/sd-class-butterflies-32'.
    The model is a diffusion model for unconditional image generation of cute butterflies.
    
    Returns:
        image: An image of a cute butterfly.
    """
    pipeline = DDPMPipeline.from_pretrained('clp/sd-class-butterflies-32')
    image = pipeline().images[0]
    return image

# test_function_code --------------------

def test_generate_butterfly_image():
    """
    This function tests the 'generate_butterfly_image' function by generating an image and checking its type.
    """
    image = generate_butterfly_image()
    assert isinstance(image, type(None)), 'The function should return an image.'

# call_test_function_code --------------------

test_generate_butterfly_image()