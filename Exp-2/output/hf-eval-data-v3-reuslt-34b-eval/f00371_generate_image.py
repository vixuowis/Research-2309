# function_import --------------------

import os
from diffusers import DDPMPipeline

# function_code --------------------

def generate_image(model_id: str) -> None:
    '''
    Generate a high-quality image using a pre-trained model.

    Args:
        model_id (str): The id of the pre-trained model.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the diffusers library is not installed.
    '''
    if 'DIFFUSERS_DATA_DIR' in os.environ:
        data_dir = os.environ['DIFFUSERS_DATA_DIR']
    else:
        data_dir = os.path.join(os.getcwd(), '.data')
    
    pipeline = DDPMPipeline()
    pipeline.load_model(model_id, 'models')
    pipeline.generate_image('outputs', data_dir)

# test_function_code --------------------

def test_generate_image():
    '''
    Test the generate_image function.

    Returns:
        str: 'All Tests Passed' if all assertions pass.
    '''
    try:
        generate_image('google/ddpm-church-256')
        assert os.path.exists('ddpm_generated_image.png')
    except Exception as e:
        print(f'Test failed with exception: {e}')
        raise e
    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_generate_image())