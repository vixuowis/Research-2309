# function_import --------------------

import os
from diffusers import DDPMPipeline

# function_code --------------------

def generate_cat_image(model_id: str = 'google/ddpm-ema-cat-256') -> None:
    """
    Generate a cat image using a pre-trained model from Hugging Face Transformers.

    Args:
        model_id (str): The ID of the pre-trained model. Default is 'google/ddpm-ema-cat-256'.

    Returns:
        None. The function saves the generated image to the current directory.

    Raises:
        ModuleNotFoundError: If the diffusers package is not installed.
    """    
    
    if not os.path.exists('ddpm_diffuser.pt'):
        # download model checkpoint
        print("Downloading DDPM model checkpoint...")
        os.system("wget https://cdn.huggingface.co/diffusers/ddpm_diffuser.pt")
        
    pipeline = DDPMPipeline(model_id)
    
    # generate image and save it
    img = pipeline('')
    img[0].save('output.png')

# test_function_code --------------------

def test_generate_cat_image():
    """
    Test the generate_cat_image function.

    Returns:
        str: 'All Tests Passed' if all assertions pass.
    """
    generate_cat_image()
    assert os.path.exists('ddpm_generated_cat_image.png'), 'Image not generated'
    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_generate_cat_image())