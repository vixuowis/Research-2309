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

    # Check that the model exists.
    try:
        import transformers  # noqa
    except ImportError as e:
        raise e("transformers must be installed to use this function.")

    if (model_id == 'clp/sd-class-butterflies-32'):
        model = DDPMPipeline.from_pretrained('clp/sd-class-butterflies-32')
    else:
        raise ValueError(f'No pre-trained model for {model_id}.')

    # Generate a butterfly image.
    result = model("I really like this butterfly.", return_image=True)

    # Save the generated image to the current directory.
    result[0].save('butterfly.png')

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