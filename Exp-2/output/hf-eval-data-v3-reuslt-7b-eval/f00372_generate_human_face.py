# function_import --------------------

from diffusers import DiffusionPipeline
import os

# function_code --------------------

def generate_human_face(model_id: str = 'google/ncsnpp-ffhq-256') -> None:
    """
    Generate a synthetic human face image using a pre-trained model.

    Args:
        model_id (str): The identifier of the pre-trained model. Default is 'google/ncsnpp-ffhq-256'.

    Returns:
        None. The function saves the generated image to a file named 'sde_ve_generated_image.png'.
    """

    # Create the Diffusion pipeline object and load the pre-trained model.
    diffuser = DiffusionPipeline(model=model_id)
    
    # Generate the synthetic image.
    generated_image = diffuser.sample()

    # Save the synthetic image to a png file.
    os.makedirs('output', exist_ok=True)
    generated_image.save(os.path.join('output','sde_ve_generated_image.png'))
    
    return None

# test_function_code --------------------

def test_generate_human_face():
    """
    Test the function generate_human_face.
    """
    generate_human_face()
    assert os.path.exists('sde_ve_generated_image.png'), 'Image not generated.'
    os.remove('sde_ve_generated_image.png')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_human_face()