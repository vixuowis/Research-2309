# requirements_file --------------------

!pip install -U diffusers PIL

# function_import --------------------

from diffusers import DDPMPipeline
import PIL.Image

# function_code --------------------

def generate_butterfly_image():
    """
    Generates an image of a butterfly using the pre-trained model from Hugging Face.

    Returns:
        PIL.Image: An image of a generated butterfly.
    """
    # Load the pre-trained diffusion model
    pipeline = DDPMPipeline.from_pretrained('utyug1/sd-class-butterflies-32')

    # Generate a butterfly image
    generated_image_tensor = pipeline().images[0]

    # Convert the tensor to PIL image
    generated_image = PIL.Image.fromarray(generated_image_tensor.numpy())
    return generated_image

# test_function_code --------------------

def test_generate_butterfly_image():
    print("Testing generate_butterfly_image function...")

    # Generate the butterfly image
    image = generate_butterfly_image()

    # Check if an image is returned
    assert isinstance(image, PIL.Image.Image), "Returned object is not a PIL image."

    # Check if the image has the right mode (RGB)
    assert image.mode == 'RGB', "Generated image does not have RGB mode."

    # Todo: Include additional image quality checks if necessary

    print("Test passed!")

# Run the test function
test_generate_butterfly_image()