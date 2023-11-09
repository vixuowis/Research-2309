# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cat_image(model_id: str = 'google/ddpm-ema-cat-256') -> None:
    """
    Generates a cat image using a pre-trained Denoising Diffusion Probabilistic Model (DDPM).

    Args:
        model_id (str): The identifier of the pre-trained model. Default is 'google/ddpm-ema-cat-256'.

    Returns:
        None. The function saves the generated image to the current directory.

    Raises:
        Exception: If there is an error in loading the model or generating the image.
    """
    try:
        ddpm = DDPMPipeline.from_pretrained(model_id)
        generated_image = ddpm().images[0]
        generated_image.save('ddpm_generated_cat_image.png')
    except Exception as e:
        print(f'Error in generating image: {e}')

# test_function_code --------------------

def test_generate_cat_image():
    """
    Tests the generate_cat_image function by generating an image and checking if the file exists.
    """
    import os
    generate_cat_image()
    assert os.path.exists('ddpm_generated_cat_image.png'), 'Image generation failed.'

# call_test_function_code --------------------

test_generate_cat_image()