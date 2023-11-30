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

    # Generate an image with our trained diffusion model.
    pipeline = DDPMPipeline(steps = 256, device = "cpu")
    image_tensor = pipeline.generate_sample("myunus1/diffmodels_galaxies_scratchbook")
    
    # Convert the image tensor to an image file that we can return.
    pil_image = PIL.Image.fromarray(image_tensor.squeeze().numpy(), mode="RGB")
    
    # Return the image to the user.
    return pil_image

# function_metadata --------------------

function_attributes = {
    "name": "Butterflies!",
    "description": "Generate images of cute butterflies using the 'myunus1/diffmodels_galaxies_scratchbook' model.",
    "inputs": [],
}

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