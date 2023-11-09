# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cat_image():
    """
    Generate a cat image using the pre-trained model 'google/ddpm-ema-cat-256'.

    This function uses the DDPMPipeline from the diffusers package to load the pre-trained model and generate a cat image.
    The generated image is then saved to the file 'ddpm_generated_cat_image.png'.

    Returns:
        None
    """
    ddpm = DDPMPipeline.from_pretrained('google/ddpm-ema-cat-256')
    image = ddpm().images[0]
    image.save('ddpm_generated_cat_image.png')

# test_function_code --------------------

def test_generate_cat_image():
    """
    Test the function generate_cat_image.

    This function calls the generate_cat_image function and checks if the file 'ddpm_generated_cat_image.png' exists.
    """
    import os
    generate_cat_image()
    assert os.path.exists('ddpm_generated_cat_image.png')

# call_test_function_code --------------------

test_generate_cat_image()