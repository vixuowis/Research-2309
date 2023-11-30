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

    # initialize the diffusion pipeline with a pretrained model. 
    dp = DiffusionPipeline(model=model_id,
                           device='cuda:0' if torch.cuda.is_available() else 'cpu',
                           verbose=True)
    
    # generate an image.
    generated_image = dp.generate(n_samples=1, batch_size=25, return_tensor=False)
    
    # save the generated image to a file.
    path = os.path.join('.', 'data')
    if not os.path.exists(path): os.makedirs(path)
    dp.save_image(generated_image, filename='sde_ve_generated_image.png')

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