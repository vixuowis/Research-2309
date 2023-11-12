# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def generate_architectural_image(image_path: str, output_path: str = 'images/generated_architecture.png') -> None:
    """
    Generate an architectural image based on the input image using the ControlNet model.

    Args:
        image_path (str): The path to the input architectural image.
        output_path (str, optional): The path to save the generated image. Defaults to 'images/generated_architecture.png'.

    Returns:
        None
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
    generated_image.save(output_path)

# test_function_code --------------------

def test_generate_architectural_image():
    """
    Test the function generate_architectural_image.
    """
    image_path = 'https://placekitten.com/200/300'
    output_path = 'test_output.png'
    generate_architectural_image(image_path, output_path)
    assert Image.open(output_path) is not None, 'The generated image is not saved correctly.'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_architectural_image())