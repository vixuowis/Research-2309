# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_celebrity_face(model_id: str = 'google/ddpm-ema-celebahq-256') -> None:
    """
    Generates a high-quality image of a celebrity face using the Denoising Diffusion Probabilistic Models (DDPM) pipeline.

    Args:
        model_id (str): The model id of the pretrained model to use for image generation. Default is 'google/ddpm-ema-celebahq-256'.

    Returns:
        None. The function saves the generated image to the local system.
    """
    ddpm = DDPMPipeline.from_pretrained(model_id)
    created_image = ddpm().images[0]
    created_image.save('generated_celebrity_face.png')

# test_function_code --------------------

def test_generate_celebrity_face():
    """
    Tests the generate_celebrity_face function by generating an image and checking if the file was created.
    """
    import os
    generate_celebrity_face()
    assert os.path.isfile('generated_celebrity_face.png')

# call_test_function_code --------------------

test_generate_celebrity_face()