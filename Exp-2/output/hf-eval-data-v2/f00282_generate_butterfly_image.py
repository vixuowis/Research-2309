# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_image():
    """
    This function generates a butterfly image using a pre-trained model from Hugging Face Transformers.

    The model used is 'utyug1/sd-class-butterflies-32', which is a diffusion model for unconditional image generation of butterflies.

    Returns:
        generated_image: An image of a butterfly generated by the model.
    """
    pipeline = DDPMPipeline.from_pretrained('utyug1/sd-class-butterflies-32')
    generated_image = pipeline().images[0]
    return generated_image

# test_function_code --------------------

def test_generate_butterfly_image():
    """
    This function tests the 'generate_butterfly_image' function by generating an image and checking if the output is not None.
    """
    generated_image = generate_butterfly_image()
    assert generated_image is not None, 'The generated image should not be None.'

# call_test_function_code --------------------

test_generate_butterfly_image()