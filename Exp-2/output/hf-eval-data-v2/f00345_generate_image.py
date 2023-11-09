# function_import --------------------

from diffusers import StableDiffusionPipeline
import torch

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'prompthero/openjourney', save_path: str = './generated_image.png'):
    """
    Generate an image based on a given text prompt using a pre-trained model.

    Args:
        prompt (str): The text prompt describing the image to be generated.
        model_id (str, optional): The id of the pre-trained model to use. Defaults to 'prompthero/openjourney'.
        save_path (str, optional): The path where the generated image will be saved. Defaults to './generated_image.png'.

    Returns:
        None. The generated image is saved at the specified path.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    image = pipe(prompt).images[0]
    image.save(save_path)

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.

    This function does not return anything but will raise an error if the generate_image function
    is not working correctly.
    """
    prompt = 'A vintage sports car racing through a desert landscape during sunset'
    save_path = './test_image.png'
    generate_image(prompt, save_path=save_path)
    assert os.path.exists(save_path), 'Image not generated.'

# call_test_function_code --------------------

test_generate_image()