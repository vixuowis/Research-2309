# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_vintage_images():
    """
    This function generates vintage images using a pretrained model from Hugging Face Transformers.

    The model is a Denoising Diffusion Probabilistic Model fine-tuned on 3 epochs of vintage images.

    Returns:
        generated_images: A list of generated images.
    """
    pipeline = DDPMPipeline.from_pretrained('pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs')
    generated_images = pipeline().images
    return generated_images

# test_function_code --------------------

def test_generate_vintage_images():
    """
    This function tests the generate_vintage_images function.

    It asserts that the function returns a list and that the list is not empty.
    """
    generated_images = generate_vintage_images()
    assert isinstance(generated_images, list), 'The function should return a list.'
    assert len(generated_images) > 0, 'The list of generated images should not be empty.'

# call_test_function_code --------------------

test_generate_vintage_images()