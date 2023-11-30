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
    # Setup the pipeline to generate a high quality image.
    pipeline = DDPMPipeline(model_id)

    # Generate the output.
    pipeline()

# function_call --------------------

# Check if there are any pre-trained models in the model directory. If not, download and extract them. Then run generate_image for each model.
for model_dir in os.listdir('models'):
    # Make sure we're dealing with a folder and not a hidden file (i.e., .DS_Store).
    if os.path.isdir(os.path.join('models', model_dir)):
        try:
            # Generate image using this pre-trained model.
            generate_image(model_dir)
            
        except ModuleNotFoundError as e:
            print("It looks like you don't have the diffusers library installed on your system.")
            print('Please run `pip install "diffusers>=0.3" --force-reinstall` to get started.')

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