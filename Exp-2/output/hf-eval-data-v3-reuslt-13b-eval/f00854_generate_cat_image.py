# function_import --------------------

from diffusers import DDPMPipeline
import os

# function_code --------------------

def generate_cat_image(model_id: str = 'google/ddpm-ema-cat-256', output_file: str = 'ddpm_generated_cat_image.png'):
    """
    Generate a cat image using a pre-trained model from Hugging Face Transformers.

    Args:
        model_id (str): The ID of the pre-trained model. Default is 'google/ddpm-ema-cat-256'.
        output_file (str): The file path to save the generated image. Default is 'ddpm_generated_cat_image.png'.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the diffusers package is not installed.
    """

    if os.path.exists(output_file): return  # skip if file already exists
    pipeline = DDPMPipeline.from_pretrained(model_id)
    output = pipeline(prompt='A cat in a field with flowers', num_init_imgs=32, num_iterations=1000)
    
    return output.save_image('ddpm_generated_cat_image.png')

# test_function_code --------------------

def test_generate_cat_image():
    """
    Test the generate_cat_image function.
    """
    try:
        generate_cat_image()
        assert os.path.exists('ddpm_generated_cat_image.png')
    except Exception as e:
        print(f'Test failed with exception: {e}')
    else:
        print('All Tests Passed')


# call_test_function_code --------------------

test_generate_cat_image()