# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cute_butterfly_image():
    """Generate an image of a cute butterfly using the 'clp/sd-class-butterflies-32' model.

    Args:
        None

    Returns:
        image (PIL.Image.Image): An image of a cute butterfly.

    Raises:
        RuntimeError: If the model fails to generate an image.
    """
    try:
        pipeline = DDPMPipeline.from_pretrained('clp/sd-class-butterflies-32')
        image = pipeline().images[0]
        return image
    except Exception as e:
        raise RuntimeError('Failed to generate butterfly image: ' + str(e))

# test_function_code --------------------

def test_generate_cute_butterfly_image():
    print("Testing started.")

    # Test case: Check if the function returns an image
    print("Testing case [1/1] started.")
    image = generate_cute_butterfly_image()
    assert isinstance(image, Image.Image), f"Test case [1/1] failed: Expected an instance of type PIL.Image.Image, got {type(image)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_cute_butterfly_image()