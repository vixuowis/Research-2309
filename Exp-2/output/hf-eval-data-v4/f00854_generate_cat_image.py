# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cat_image(model_id):
    """
    Generate an original cat image using a pretrained DDPM model.

    Parameters:
        model_id (str): The model identifier for the pretrained DDPM model.

    Returns:
        PIL.Image: The generated image of a cat.
    """
    ddpm = DDPMPipeline.from_pretrained(model_id)
    image = ddpm().images[0]
    return image

# test_function_code --------------------

def test_generate_cat_image():
    print("Testing generate_cat_image function.")
    try:
        model_id = 'google/ddpm-ema-cat-256'
        image = generate_cat_image(model_id)

        # Test case 1: Check if the function returns an image
        assert image is not None, "Image not generated."
        print("Test case passed: Image generated.")

        # Other test cases can be added here if necessary
    except AssertionError as error:
        print(f"Test case failed: {error}")
    print("Testing finished.")

test_generate_cat_image()