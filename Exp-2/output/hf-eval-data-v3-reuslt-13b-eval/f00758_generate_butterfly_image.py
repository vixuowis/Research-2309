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
    
    # Load the model to generate butterflies from random noise.
    ddpmp = DDPMPipeline(
        "https://huggingface.co/myunus1/diffmodels_galaxies_scratchbook", 48)

    # Generate a random latent vector for the butterfly.
    butterfly_latents = ddpmp.model.sample_random_latents(batch_size=1)
    
    # Generate an image from that latent vector.
    butterfly_image = ddpmp.generate(butterfly_latents)[0]  # remove batch dimension

    return PIL.Image.fromarray((butterfly_image * 255).astype('uint8'))


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