# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_car_image(model_id: str = 'google/ddpm-cifar10-32', output_file: str = 'ddpm_generated_image.png'):
    '''
    Generate a car image using the specified pre-trained model.

    Args:
        model_id: The ID of the pre-trained model to use for image generation. Default is 'google/ddpm-cifar10-32'.
        output_file: The name of the file to save the generated image to. Default is 'ddpm_generated_image.png'.

    Returns:
        None. The generated image is saved to a file.

    Raises:
        ModuleNotFoundError: If the diffusers module is not installed.
    '''
    p = DDPMPipeline(model_id)

    p([])(output_file=output_file, temperature=0.1)


# test_function_code --------------------

def test_generate_car_image():
    '''
    Test the generate_car_image function.

    Returns:
        'All Tests Passed' if all assertions pass.
    '''
    import os

    # Test with default parameters
    generate_car_image()
    assert os.path.exists('ddpm_generated_image.png'), 'Test failed: Default output file not found.'

    # Test with custom parameters
    generate_car_image(model_id='google/ddpm-cifar10-32', output_file='custom_output.png')
    assert os.path.exists('custom_output.png'), 'Test failed: Custom output file not found.'

    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_generate_car_image())