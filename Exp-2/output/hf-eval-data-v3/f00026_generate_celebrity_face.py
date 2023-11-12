# function_import --------------------

import os
from diffusers import DDPMPipeline

# function_code --------------------

def generate_celebrity_face(model_id: str) -> None:
    '''
    Generates a high-quality image of a celebrity face using the Denoising Diffusion Probabilistic Models (DDPM) pipeline.

    Args:
        model_id (str): The model id of the pretrained model to be used for image generation. For example, 'google/ddpm-ema-celebahq-256'.

    Returns:
        None. The function saves the generated image to the local system with the filename 'generated_celebrity_face.png'.
    '''
    ddpm = DDPMPipeline.from_pretrained(model_id)
    created_image = ddpm().images[0]
    created_image.save('generated_celebrity_face.png')

# test_function_code --------------------

def test_generate_celebrity_face():
    '''
    Tests the function generate_celebrity_face.
    '''
    # Test case 1: Using the model 'google/ddpm-ema-celebahq-256'
    generate_celebrity_face('google/ddpm-ema-celebahq-256')
    assert os.path.exists('generated_celebrity_face.png'), 'Test case 1 failed'
    # Test case 2: Using the model 'google/ddpm-ema-celebahq-512'
    generate_celebrity_face('google/ddpm-ema-celebahq-512')
    assert os.path.exists('generated_celebrity_face.png'), 'Test case 2 failed'
    # Test case 3: Using the model 'google/ddpm-ema-celebahq-1024'
    generate_celebrity_face('google/ddpm-ema-celebahq-1024')
    assert os.path.exists('generated_celebrity_face.png'), 'Test case 3 failed'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_celebrity_face())