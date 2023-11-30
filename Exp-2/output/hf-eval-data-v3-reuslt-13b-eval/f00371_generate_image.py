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
    # Setup --------------------

    if os.path.exists(f'data/{model_id}'): raise RuntimeError('Model has already been trained')
    
    model = DDPMPipeline.load(f'models/pre-trained/{model_id}', device='cpu')
    model.generate_image(f'data/{model_id}/image.png')
    print()

    # Teardown --------------------
    
    del model

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