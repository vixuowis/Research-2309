# function_import --------------------

import torch
from controlnet_aux import MLSDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_toy_robot_image(prompt: str, seed: int = 0):
    '''
    Generate an image of a toy robot using a pretrained ControlNet model.

    Args:
        prompt: A string that describes the image to generate.
        seed: An integer that sets the random seed for image generation.

    Returns:
        A PIL Image object of the generated image.
    '''
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    initial_image = None
    control_image = processor(initial_image)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(seed)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    return image

# test_function_code --------------------

def test_generate_toy_robot_image():
    '''
    Test the generate_toy_robot_image function.
    '''
    prompt = 'toy robot'
    image = generate_toy_robot_image(prompt)
    assert image is not None, 'The function did not return an image.'
    assert isinstance(image, Image.Image), 'The function did not return a PIL Image object.'

# call_test_function_code --------------------

test_generate_toy_robot_image()