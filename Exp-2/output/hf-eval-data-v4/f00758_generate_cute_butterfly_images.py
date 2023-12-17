# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cute_butterfly_images(model_name='myunus1/diffmodels_galaxies_scratchbook'):
    # Load the diffusion model from Hugging Face Hub
    pipeline = DDPMPipeline.from_pretrained(model_name)
    # Generate an image of a cute butterfly
    generated_data = pipeline()
    image = generated_data.images[0]
    # Return the generated image
    return image

# test_function_code --------------------

def test_generate_cute_butterfly_images():
    print("Testing generate_cute_butterfly_images function.")
    # Generate an image
    generated_image = generate_cute_butterfly_images()
    # Test case 1: Check if the function returns an image
    assert isinstance(generated_image, Image.Image), "The function should return an instance of PIL.Image.Image."
    print("Test case [1/1] passed.")
    print("Testing completed.")

# Run the test function
test_generate_cute_butterfly_images()