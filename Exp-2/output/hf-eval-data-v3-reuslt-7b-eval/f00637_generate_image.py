# function_import --------------------

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import os

# function_code --------------------

def generate_image(prompt: str, model_id: str = 'stabilityai/stable-diffusion-2-1-base', output_file: str = 'output.png'):
    """
    Generate an image based on the provided text prompt using the StableDiffusionPipeline.

    Args:
        prompt (str): The text prompt to generate the image from.
        model_id (str, optional): The model id to use for the generation. Defaults to 'stabilityai/stable-diffusion-2-1-base'.
        output_file (str, optional): The file to save the generated image to. Defaults to 'output.png'.

    Returns:
        None
    """
    
    print(f"Generating Image with prompt \"{prompt}\"")

    # Load model
    model = StableDiffusionPipeline.from_pretrained(model_id)

    # Generate a sample
    output = model.sample(text=prompt, clip=True, return_images=False)
    print("Successfully generated image")
    
    # Save image to file
    torchvision.utils.save_image((output), output_file)
    
    print(f"Image saved as {output_file}")

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.
    """
    generate_image('a lighthouse on a foggy island', output_file='test_output.png')
    assert os.path.exists('test_output.png'), 'Test failed: Image file not found.'
    os.remove('test_output.png')
    print('All Tests Passed')


# call_test_function_code --------------------

test_generate_image()