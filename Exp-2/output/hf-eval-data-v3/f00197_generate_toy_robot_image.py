# function_import --------------------

import torch
from controlnet_aux import MLSDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

# function_code --------------------

def generate_toy_robot_image(prompt: str, num_inference_steps: int = 30, seed: int = 0, initial_image = None):
    """
    Generate an image of a toy robot using a pretrained ControlNet model.

    Args:
        prompt (str): Text prompt to control the image generation, e.g. 'toy robot'.
        num_inference_steps (int, optional): Number of inference steps. Default is 30.
        seed (int, optional): Random seed for image generation. Default is 0.
        initial_image (optional): Initial image to start the generation. If not provided, a random image will be used.

    Returns:
        PIL.Image: Generated image.
    """
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    control_image = processor(initial_image)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(seed)
    image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator, image=control_image).images[0]
    return image

# test_function_code --------------------

def test_generate_toy_robot_image():
    """
    Test the function generate_toy_robot_image.
    """
    image1 = generate_toy_robot_image('toy robot')
    assert image1 is not None
    assert isinstance(image1, torch.Tensor)
    image2 = generate_toy_robot_image('toy robot', num_inference_steps=10, seed=1)
    assert image2 is not None
    assert isinstance(image2, torch.Tensor)
    assert image1.shape == image2.shape
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_toy_robot_image())