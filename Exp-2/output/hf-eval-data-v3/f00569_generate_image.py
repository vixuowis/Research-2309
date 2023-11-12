# function_import --------------------

import torch
from pathlib import Path
from diffusers.utils import load_image
from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_image(prompt: str, num_inference_steps: int = 30, seed: int = 0, output_path: str = 'generated_image.png'):
    '''
    Generate an image based on a text prompt using a pre-trained ControlNetModel.

    Args:
        prompt (str): The text prompt to generate the image from.
        num_inference_steps (int, optional): The number of inference steps. Defaults to 30.
        seed (int, optional): The seed for the random number generator. Defaults to 0.
        output_path (str, optional): The path to save the generated image. Defaults to 'generated_image.png'.

    Returns:
        None
    '''
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(seed)
    generated_image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
    generated_image.save(output_path)

# test_function_code --------------------

def test_generate_image():
    '''
    Test the generate_image function.
    '''
    generate_image('A magical forest with unicorns and a rainbow.')
    assert Path('generated_image.png').exists(), 'Image not generated'
    generate_image('A city skyline at sunset.', output_path='city_sunset.png')
    assert Path('city_sunset.png').exists(), 'Image not generated'
    generate_image('A serene beach with palm trees.', output_path='beach.png')
    assert Path('beach.png').exists(), 'Image not generated'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_image()