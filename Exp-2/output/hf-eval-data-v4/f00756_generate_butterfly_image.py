# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_image():
    """
    Generate an image of a butterfly using a pretrained model.

    Returns:
        A PIL Image object of the generated butterfly image.
    """
    butterfly_generator = DDPMPipeline.from_pretrained('ocariz/butterfly_200')
    butterfly_image = butterfly_generator().images[0]
    return butterfly_image

# test_function_code --------------------

def test_generate_butterfly_image():
    print("Testing started.")
    # No dataset required, as model generates images condition-free

    # Testing if the function generates an image
    print("Testing case [1/1] started.")
    generated_image = generate_butterfly_image()
    assert generated_image is not None, f"Test case [1/1] failed: No image was generated."
    assert isinstance(generated_image, Image.Image), f"Test case [1/1] failed: Generated image is not a PIL Image object."
    print("Testing finished.")

# Run the test
if __name__ == '__main__':
    test_generate_butterfly_image()