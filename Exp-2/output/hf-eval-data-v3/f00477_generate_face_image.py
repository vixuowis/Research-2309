# function_import --------------------

from diffusers import DiffusionPipeline
import os

# function_code --------------------

def generate_face_image(model_id: str, image_path: str) -> None:
    '''
    Generate a high-resolution image of a human face using a pre-trained model and save it to a file.

    Args:
        model_id (str): The ID of the pre-trained model to use for image generation.
        image_path (str): The path where the generated image will be saved.

    Returns:
        None
    '''
    sde_ve = DiffusionPipeline.from_pretrained(model_id)
    image = sde_ve()[0]
    image.save(image_path)

# test_function_code --------------------

def test_generate_face_image():
    '''
    Test the generate_face_image function.
    '''
    model_id = 'google/ncsnpp-celebahq-256'
    image_path = 'test_generated_face.png'
    generate_face_image(model_id, image_path)
    assert os.path.exists(image_path), 'Image file does not exist.'
    os.remove(image_path)
    assert not os.path.exists(image_path), 'Failed to delete the image file.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_face_image()