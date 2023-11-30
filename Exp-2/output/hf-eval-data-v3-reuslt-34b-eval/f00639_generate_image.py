# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch
import os

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'dreamlike-art/dreamlike-photoreal-2.0', device: str = 'cuda') -> None:
    """
    Generate an image based on the given text prompt using a pre-trained model.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str, optional): The ID of the pre-trained model to use. Defaults to 'dreamlike-art/dreamlike-photoreal-2.0'.
        device (str, optional): The device to run the model on. Defaults to 'cuda'.

    Returns:
        None. The function saves the generated image as 'result.png' in the current directory.
    """    
    pipeline = StableDiffusionPipeline(
        diffusion_model_output_size=64,
        model_path=os.path.join('pretrained', f'{model_id}.pt'), # https://huggingface.co/models?other=dreamlike-art
        device=device)
    pipeline(prompt, seed=0)
    
    os.makedirs('outputs', exist_ok=True)
    torch.save(pipeline.image, 'outputs/result.png')

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.
    """
    generate_image('astronaut playing guitar in space')
    assert os.path.exists('result.png'), 'Image not generated'
    os.remove('result.png')
    generate_image('a cat sitting on a tree')
    assert os.path.exists('result.png'), 'Image not generated'
    os.remove('result.png')
    generate_image('a beautiful sunset over the ocean')
    assert os.path.exists('result.png'), 'Image not generated'
    os.remove('result.png')
    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_generate_image())