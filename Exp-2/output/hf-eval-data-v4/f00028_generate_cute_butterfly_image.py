# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cute_butterfly_image(model_id:str = 'clp/sd-class-butterflies-32') -> Image:
    """
    Generate an image of a cute butterfly using a pre-trained diffusion model.

    Args:
        model_id (str): The identifier for the pre-trained model.

    Returns:
        An image object of the generated butterfly.
    """
    pipeline = DDPMPipeline.from_pretrained(model_id)
    image = pipeline().images[0]
    return image

# test_function_code --------------------

def test_generate_cute_butterfly_image():
    print("Testing generate_cute_butterfly_image started.")

    # Test case: Check if returned object is an image
    result = generate_cute_butterfly_image()
    assert isinstance(result, Image), f"The returned object is not an image: {type(result)}"

    print("Testing generate_cute_butterfly_image finished.")

test_generate_cute_butterfly_image()