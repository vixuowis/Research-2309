# function_import --------------------

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'stabilityai/stable-diffusion-2-1-base', output_file: str = 'output.png'):
    """
    Generate an image based on a text prompt using the StableDiffusionPipeline from Hugging Face.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str, optional): The model id to use for the generation. Defaults to 'stabilityai/stable-diffusion-2-1-base'.
        output_file (str, optional): The file to save the generated image to. Defaults to 'output.png'.

    Returns:
        None. The function saves the generated image to the specified output file.
    """
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    image = pipe(prompt).images[0]
    image.save(output_file)

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.

    The function does not return anything, so the test will pass if the function runs without raising an exception.
    """
    try:
        generate_image('a lighthouse on a foggy island')
    except Exception as e:
        assert False, f'Exception occurred: {e}'

# call_test_function_code --------------------

test_generate_image()