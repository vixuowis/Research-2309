# function_import --------------------

from diffusers import StableDiffusionInpaintPipeline
import torch

# function_code --------------------

def generate_image_from_text(prompt: str, save_path: str = 'generated_image.png'):
    """
    Generate an image based on the given text prompt using the StableDiffusionInpaintPipeline from the diffusers package.

    Args:
        prompt (str): The text description of the desired image.
        save_path (str): The path where the generated image will be saved. Default is 'generated_image.png'.

    Returns:
        None. The generated image is saved to the specified path.
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16)
    pipe.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    generated_image = pipe(prompt=prompt).images[0]
    generated_image.save(save_path)

# test_function_code --------------------

def test_generate_image_from_text():
    """
    Test the function generate_image_from_text.

    The function is considered passed if no exception is raised during the execution.
    """
    test_prompt = 'A modern living room with a fireplace and a large window overlooking a forest.'
    test_save_path = 'test_image.png'
    try:
        generate_image_from_text(test_prompt, test_save_path)
    except Exception as e:
        assert False, f'Exception occurred: {e}'

# call_test_function_code --------------------

test_generate_image_from_text()