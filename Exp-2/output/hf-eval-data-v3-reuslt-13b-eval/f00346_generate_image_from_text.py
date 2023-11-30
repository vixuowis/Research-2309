# function_import --------------------

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import os

# function_code --------------------

def generate_image_from_text(prompt: str, model_id: str = 'stabilityai/stable-diffusion-2-1', save_path: str = 'generated_image.png'):
    """
    Generate an image based on the given text description using the StableDiffusionPipeline model.

    Args:
        prompt (str): The text description of the scene.
        model_id (str, optional): The id of the pre-trained model. Defaults to 'stabilityai/stable-diffusion-2-1'.
        save_path (str, optional): The path to save the generated image. Defaults to 'generated_image.png'.

    Returns:
        None
    """
    if not torch.cuda.is_available():
        raise Exception('GPU is required to run function.')
    model = StableDiffusionPipeline()
    model.load(model_id, 'cpu')
    
    # Generate the images using diffusion
    model.generate_images_from_text(prompt)
    output = model.decode_generated_output()[0]
    img = Image.open(output.path)
    img.save(save_path, 'PNG')
    os.remove(output.path)
    
# function_code --------------------


# test_function_code --------------------

def test_generate_image_from_text():
    """
    Test the function generate_image_from_text.

    Returns:
        str: 'All Tests Passed' if all tests pass, otherwise the error message.
    """
    try:
        generate_image_from_text('a scene of a magical forest with fairies and elves')
        assert os.path.exists('generated_image.png')
        os.remove('generated_image.png')
        return 'All Tests Passed'
    except Exception as e:
        return str(e)


# call_test_function_code --------------------

print(test_generate_image_from_text())