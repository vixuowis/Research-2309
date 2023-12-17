# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_gallery_image(model_id='google/ddpm-church-256'):
    """
    Generate a 256x256 image using a Denoising Diffusion Probabilistic Model.

    Args:
        model_id (str): The model ID of the pre-trained image generation model.

    Returns:
        PIL.Image: The generated image in PIL format.
    """
    # Load the pre-trained image generation model
    ddpm = DDPMPipeline.from_pretrained(model_id)
    
    # Generate the image
    image = ddpm().images[0]

    return image

# test_function_code --------------------

def test_generate_gallery_image():
    print("Testing started.")

    # Test case 1: Default model ID
    print("Testing default model ID [1/1] started.")
    image = generate_gallery_image()
    assert image.size == (256, 256), f"Test case failed: Expected image size (256, 256), got {image.size}"
    print("Testing finished.")