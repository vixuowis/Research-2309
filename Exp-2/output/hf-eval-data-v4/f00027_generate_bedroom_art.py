# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_bedroom_art(model_name='johnowhitaker/sd-class-wikiart-from-bedrooms'):
    """
    Generate an image of bedroom art using a pretrained diffusion model.

    Parameters:
        model_name (str): Identifier for the pretrained model.

    Returns:
        PIL.Image: The generated image of bedroom art.
    """
    # Load the pretrained diffusion model
    pipeline = DDPMPipeline.from_pretrained(model_name)
    # Generate the image
    generated_image = pipeline().images[0]
    return generated_image

# test_function_code --------------------

def test_generate_bedroom_art():
    print("Testing generate_bedroom_art function.")
    # Test case 1: Ensure that an image is generated
    image = generate_bedroom_art()
    assert image is not None, "Test case failed: No image was generated."

    # Test case 2: Ensure that the generated image has the right type
    assert isinstance(image, PIL.Image.Image), "Test case failed: Generated object is not an image."

    # Test case 3: Optionally, display the image if running interactively (commented out by default)
    # image.show()

    print("All tests passed!")

# Run the test for generate_bedroom_art function
# Comment this out if you only want to define the test
# test_generate_bedroom_art()