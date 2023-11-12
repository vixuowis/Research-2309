# function_import --------------------

from diffusers import DDPMPipeline
import os

# function_code --------------------

def generate_insect_image(model_name: str) -> None:
    '''
    Generate an insect image using a pretrained model.

    Args:
        model_name (str): The name of the pretrained model.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the diffusers package is not installed.
        Exception: If there is an error in generating the image.
    '''
    try:
        pipeline = DDPMPipeline.from_pretrained(model_name)
        generated_image = pipeline().images[0]
        generated_image.save('insect_image.png')
    except ModuleNotFoundError as e:
        print('Please install the diffusers package.')
        raise e
    except Exception as e:
        print('An error occurred in generating the image.')
        raise e

# test_function_code --------------------

def test_generate_insect_image():
    '''
    Test the generate_insect_image function.

    Returns:
        str: A message indicating that all tests passed.
    '''
    try:
        generate_insect_image('schdoel/sd-class-AFHQ-32')
        assert os.path.exists('insect_image.png'), 'The image file does not exist.'
        return 'All Tests Passed'
    except Exception as e:
        return str(e)

# call_test_function_code --------------------

if __name__ == '__main__':
    print(test_generate_insect_image())