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
        
        # load the pretrained model
        model = DDPMPipeline(model_name)

        # generate the insect image
        output = model.sample_images()

        # save the generated image
        output[0].save('output.png')

    except ModuleNotFoundError:
        
        raise ModuleNotFoundError('The package "diffusers" is required to run this function.')
    
    except Exception as e:
        
        raise Exception(e)
    
# function_code --------------------

if __name__ == '__main__':
    
    generate_insect_image('2021-08-30 23:54:49.666704')

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