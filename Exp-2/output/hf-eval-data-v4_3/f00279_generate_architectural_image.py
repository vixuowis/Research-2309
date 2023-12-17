# requirements_file --------------------

import subprocess

requirements = ["Pillow", "diffusers", "torch", "controlnet_aux"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def generate_architectural_image(image_path):
    """Generate an architectural image using a ControlNet model.

    Args:
        image_path (str): The file path to the input architectural image.

    Returns:
        Image: The generated architectural image.

    Raises:
        FileNotFoundError: If the image_path does not lead to a valid file.
    """
    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    image = load_image(image_path)
    image = mlsd(image)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    generated_image = pipe(image, num_inference_steps=20).images[0]
    
    return generated_image

# test_function_code --------------------

def test_generate_architectural_image():
    print("Testing started.")
    image_path = 'https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/images/room.png'
    
    # Test case 1: Check if function returns an image
    print("Testing case [1/1] started.")
    generated_image = generate_architectural_image(image_path)
    assert isinstance(generated_image, Image.Image), f"Test case [1/1] failed: Expected output to be an instance of PIL.Image.Image, got {type(generated_image).__name__}".
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_architectural_image()