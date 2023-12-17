# requirements_file --------------------

!pip install -U diffusers PIL

# function_import --------------------

from diffusers import DDPMPipeline
import PIL

# function_code --------------------

def generate_butterfly_image(save_path='cute_butterfly_image.png'):
    """
    Generate a cute butterfly image using a pre-trained diffusion model and save it to the specified path.
    
    Args:
        save_path (str): Location where the generated image will be saved. Defaults to 'cute_butterfly_image.png'.

    Returns:
        PIL.Image: The generated butterfly image.
    """
    # Load the pre-trained model from Hugging Face
    pipeline = DDPMPipeline.from_pretrained('clp/sd-class-butterflies-32')

    # Generate the image
    image = pipeline().images[0]

    # Save the image
    image.save(save_path)

    # Return the image as a PIL Image object in case further manipulation is needed
    return image

# test_function_code --------------------

def test_generate_butterfly_image():
    print("Testing generate_butterfly_image function.")

    # Test 1: Check if the image is saved and is a PNG file
    generated_image = generate_butterfly_image('test_butterfly_image.png')
    assert os.path.exists('test_butterfly_image.png'), "The image was not saved to the specified path."
    assert 'test_butterfly_image.png'.endswith('.png'), "The saved image is not in PNG format."

    # Test 2: Check if the returned image is a PIL Image object
    assert isinstance(generated_image, PIL.Image.Image), "The function did not return a PIL Image object."

    # Cleanup the test
    if os.path.exists('test_butterfly_image.png'):
        os.remove('test_butterfly_image.png')

    print("All tests passed!"),

    # Call the test function
    test_generate_butterfly_image()