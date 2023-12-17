# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cat_image(model_id='google/ddpm-ema-cat-256'):
    """
    Generate a 256x256 image of a cat using a pre-trained Denoising Diffusion Probabilistic Model.

    Args:
        model_id (str): The identifier of the pre-trained model. Default is 'google/ddpm-ema-cat-256'.

    Returns:
        PIL.Image: The generated cat image.
    """
    # Load the pre-trained DDPM model
    ddpm = DDPMPipeline.from_pretrained(model_id)

    # Generate the cat image
    generated_image = ddpm().images[0]

    return generated_image

# test_function_code --------------------

def test_generate_cat_image():
    print("Testing generate_cat_image function.")

    # Test case: Generate a cat image
    cat_image = generate_cat_image()
    assert cat_image.size == (256, 256), f"Test case failed: Expected image size (256, 256), got {cat_image.size}"

    print("All test cases passed for generate_cat_image function.")

# Run the test function
test_generate_cat_image()
