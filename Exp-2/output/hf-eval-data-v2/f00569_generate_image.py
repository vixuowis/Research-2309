# function_import --------------------

import torch
from pathlib import Path
from diffusers.utils import load_image
from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_image(prompt: str, num_inference_steps: int = 30, seed: int = 0, save_path: str = 'generated_image.png') -> None:
    """
    Generate an image based on a text prompt using a pre-trained ControlNetModel.

    Args:
        prompt (str): The text prompt to generate the image from.
        num_inference_steps (int, optional): The number of inference steps to use. Defaults to 30.
        seed (int, optional): The seed for the random number generator. Defaults to 0.
        save_path (str, optional): The path to save the generated image. Defaults to 'generated_image.png'.

    Returns:
        None
    """
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(seed)
    generated_image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
    generated_image.save(save_path)

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.

    Raises:
        AssertionError: If the function does not generate an image file at the specified path.
    """
    test_prompt = 'A magical forest with unicorns and a rainbow.'
    test_save_path = 'test_generated_image.png'
    generate_image(test_prompt, save_path=test_save_path)
    assert Path(test_save_path).is_file(), 'The image file was not generated.'
    Path(test_save_path).unlink()  # Clean up the test image file.

# call_test_function_code --------------------

test_generate_image()