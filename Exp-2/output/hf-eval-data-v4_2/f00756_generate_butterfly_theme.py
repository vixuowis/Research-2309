# requirements_file --------------------

!pip install -U diffusers pillow

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_theme():
    """
    Generate a butterfly image to suggest as a theme for the mobile application.

    Args:
        None

    Returns:
        PIL.Image: An image of a butterfly.
    
    Raises:
        ValueError: If the model fails to generate an image.
    """
    try:
        butterfly_generator = DDPMPipeline.from_pretrained('ocariz/butterfly_200')
        butterfly_image = butterfly_generator().images[0]
        return butterfly_image
    except Exception as e:
        raise ValueError('Failed to generate butterfly image') from e

# test_function_code --------------------

def test_generate_butterfly_theme():
    print("Testing started.")

    # There is no dataset loading required for this functionality

    # Test case 1: Check if the function returns an image
    print("Testing case [1/1] started.")
    result = generate_butterfly_theme()
    assert isinstance(result, Image.Image), f"Test case [1/1] failed: Expected result type 'PIL.Image.Image', got '{type(result)}'."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_butterfly_theme()