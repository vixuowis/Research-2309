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
    if not os.path.exists('./diffusers'):
        raise ModuleNotFoundError("You must install 'diffuser' before you can call this function")
    
    # Load a pre-trained model from Hugging Face Transformers
    # This example uses the EMA (Exponential Moving Average) model trained with 10M steps.
    diffuser = DDPMPipeline.from_pretrained(model_id).to('cuda')
    
    # Generate an image
    img = diffuser.generate()[0]
    img = (img + 1) / 2

    # Save the generated image to the current directory
    filename = os.path.join(os.getcwd(), "generated_cat_image.png")
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img.save('generated_cat_image.png')

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