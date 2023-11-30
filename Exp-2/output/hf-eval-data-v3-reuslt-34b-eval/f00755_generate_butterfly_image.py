# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_butterfly_image(model_id: str = 'clp/sd-class-butterflies-32') -> None:
    """
    Generate a butterfly image using a pre-trained model from Hugging Face Transformers.

    Args:
        model_id (str): The id of the pre-trained model. Default is 'clp/sd-class-butterflies-32'.

    Returns:
        None. The function saves the generated image to the current directory.

    Raises:
        ModuleNotFoundError: If the diffusers package is not installed.
    """

    # Create pipeline object and generate an image using it

    pipe = DDPMPipeline(text='butterflies', model_id=model_id)
    img = pipe()
    
    # Save generated image to current directory
    
    file_name = 'butterfly_image.png'
    img.save(file_name)

# test_function_code --------------------

def test_generate_butterfly_image():
    """
    Test the generate_butterfly_image function.
    """
    try:
        generate_butterfly_image()
        print('Test passed.')
    except Exception as e:
        print(f'Test failed. {str(e)}')


# call_test_function_code --------------------

test_generate_butterfly_image()