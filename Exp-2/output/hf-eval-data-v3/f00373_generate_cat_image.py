# function_import --------------------

import os
from diffusers import DDPMPipeline

# function_code --------------------

def generate_cat_image(model_id: str, save_path: str) -> None:
    '''
    Generate a cat image using a pre-trained model and save it to a specified path.

    Args:
        model_id (str): The id of the pre-trained model to use for image generation.
        save_path (str): The path where the generated image will be saved.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the diffusers module is not installed.
        Exception: If there is an error in generating the image or saving it.
    '''
    try:
        ddpm = DDPMPipeline.from_pretrained(model_id)
        image = ddpm().images[0]
        image.save(save_path)
    except ModuleNotFoundError as e:
        print('Please install the diffusers module.')
        raise e
    except Exception as e:
        print('An error occurred in generating the image or saving it.')
        raise e

# test_function_code --------------------

def test_generate_cat_image():
    '''
    Test the generate_cat_image function.
    '''
    try:
        # Test with a valid model id and save path
        generate_cat_image('google/ddpm-ema-cat-256', 'cat_character_image.png')
        assert os.path.exists('cat_character_image.png')

        # Test with an invalid model id
        try:
            generate_cat_image('invalid_model_id', 'cat_character_image.png')
        except Exception:
            pass

        # Test with an invalid save path
        try:
            generate_cat_image('google/ddpm-ema-cat-256', '/invalid/path/cat_character_image.png')
        except Exception:
            pass

        print('All Tests Passed')
    except AssertionError:
        print('Test Failed')

# call_test_function_code --------------------

test_generate_cat_image()