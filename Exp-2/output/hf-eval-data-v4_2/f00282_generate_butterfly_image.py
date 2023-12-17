# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_image(model_name):
    """
    Generate an image of a butterfly using a pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model to use for image generation.

    Returns:
        PIL.Image.Image: The generated image of a butterfly.

    Raises:
        ValueError: If the pretrained_model name is not provided or is invalid.
    """
    if not model_name:
        raise ValueError('The pretrained_model name must be provided.')

    try:
        pipeline = DDPMPipeline.from_pretrained(model_name)
        generated_image = pipeline().images[0]
        return generated_image
    except Exception as e:
        raise ValueError(f'Error generating the butterfly image: {e}')

# test_function_code --------------------

def test_generate_butterfly_image():
    print("Testing started.")
    
    # There is no dataset to load as we are generating images, not using an existing dataset.

    # Test case 1: Correct model name
    print("Testing case [1/2] started.")
    try:
        image = generate_butterfly_image('utyug1/sd-class-butterflies-32')
        assert image is not None, "Test case [1/2] failed: The function did not return an image."
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"

    # Test case 2: Incorrect model name
    print("Testing case [2/2] started.")
    try:
        generate_butterfly_image('')
        assert False, "Test case [2/2] failed: The function did not raise a ValueError with an empty model name."
    except ValueError as e:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_butterfly_image()