# function_import --------------------

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from controlnet_aux import MLSDdetector
import torch
from PIL import Image

# function_code --------------------

def generate_image(prompt: str, image_path: str, output_path: str) -> None:
    '''
    Generate an image based on the given prompt using a pre-trained ControlNetModel.

    Args:
        prompt (str): The text prompt to generate the image from.
        image_path (str): The path to the base image file.
        output_path (str): The path to save the generated image.

    Returns:
        None
    '''
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_mlsd', torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )

    with Image.open(image_path) as image:
        control_image = processor(image)
        generated_image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0), image=control_image).images[0]
        generated_image.save(output_path)

# test_function_code --------------------

def test_generate_image():
    '''
    Test the generate_image function.
    '''
    generate_image('luxury living room with a fireplace', 'test_images/base_image.png', 'test_images/generated_image.png')
    assert Image.open('test_images/generated_image.png') is not None

    generate_image('modern kitchen', 'test_images/base_image.png', 'test_images/generated_image.png')
    assert Image.open('test_images/generated_image.png') is not None

    generate_image('cozy bedroom', 'test_images/base_image.png', 'test_images/generated_image.png')
    assert Image.open('test_images/generated_image.png') is not None

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_image())