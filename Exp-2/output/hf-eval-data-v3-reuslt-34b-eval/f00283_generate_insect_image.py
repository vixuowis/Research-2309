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
    
    # validate parameters
    valid_model_names = ['ddpm', 'ddpm_wo_noise']
    if model_name not in valid_model_names:
        raise Exception('Invalid value passed to "model_name".')
    
    try:
        from diffusers import DDPMPipeline
        
        pipeline = DDPMPipeline(f'{os.getcwd()}/models/{model_name}.ckpt')
        pipeline.generate(n=1)
    
    except ModuleNotFoundError as e:
        raise e('The "diffusers" package is not installed.')
        
    except Exception as e:
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