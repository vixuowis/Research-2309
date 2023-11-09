# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def generate_architectural_image(image_path: str) -> Image:
    """
    Generate an architectural image using the ControlNet model from Hugging Face.

    Args:
        image_path (str): The path to the input architectural image.

    Returns:
        Image: The generated architectural image.
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
    """
    Test the generate_architectural_image function.
    """
    image_path = 'https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/images/room.png'
    generated_image = generate_architectural_image(image_path)
    assert isinstance(generated_image, Image), 'The output should be an instance of PIL.Image.'

# call_test_function_code --------------------

test_generate_architectural_image()