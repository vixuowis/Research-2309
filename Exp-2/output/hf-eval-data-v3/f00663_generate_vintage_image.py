# function_import --------------------

import os
from diffusers import DDPMPipeline

# function_code --------------------

def generate_vintage_image(model_name: str, output_file: str) -> None:
    """
    Generate a vintage image using a pre-trained model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pre-trained model.
        output_file (str): The path to the output file where the generated image will be saved.

    Returns:
        None
    """
    pipeline = DDPMPipeline.from_pretrained(model_name)
    vintage_image = pipeline().images[0]
    vintage_image.save(output_file)

# test_function_code --------------------

def test_generate_vintage_image():
    """
    Test the generate_vintage_image function.
    """
    model_name = 'pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs'
    output_file = 'test_vintage_image.png'
    generate_vintage_image(model_name, output_file)
    assert os.path.exists(output_file), 'Test failed: Image file not found.'
    os.remove(output_file)
    print('All Tests Passed')

# call_test_function_code --------------------

test_generate_vintage_image()