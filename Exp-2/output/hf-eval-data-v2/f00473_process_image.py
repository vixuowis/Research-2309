# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def process_image(input_image_path: str, output_image_path: str, num_inference_steps: int = 20):
    """
    Process an image by detecting straight lines and controlling the diffusion models in the image's diffusion process.

    Args:
        input_image_path (str): The path to the input image.
        output_image_path (str): The path to save the processed image.
        num_inference_steps (int, optional): The number of inference steps. Defaults to 20.

    Returns:
        None
    """
    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    image = load_image(input_image_path)
    image = mlsd(image)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    processed_image = pipe(image, num_inference_steps=num_inference_steps).images[0]
    processed_image.save(output_image_path)

# test_function_code --------------------

def test_process_image():
    """
    Test the process_image function.
    """
    process_image('test_input_image.png', 'test_output_image.png')
    assert Image.open('test_output_image.png') is not None

# call_test_function_code --------------------

test_process_image()