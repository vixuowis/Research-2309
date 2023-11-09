# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def transform_floor_plan(input_image_path: str, output_image_path: str, num_inference_steps: int = 20):
    """
    Transforms a floor plan image into a simplified straight line drawing.

    Args:
        input_image_path (str): The path to the input floor plan image.
        output_image_path (str): The path to save the output simplified image.
        num_inference_steps (int, optional): The number of inference steps to use in the pipeline. Defaults to 20.

    Returns:
        None
    """
    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    floor_plan_img = load_image(input_image_path)
    floor_plan_img = mlsd(floor_plan_img)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    result_img = pipe(floor_plan_img, num_inference_steps=num_inference_steps).images[0]
    result_img.save(output_image_path)

# test_function_code --------------------

def test_transform_floor_plan():
    """
    Tests the transform_floor_plan function.
    """
    input_image_path = 'test_floor_plan.png'
    output_image_path = 'test_floor_plan_simplified.png'
    num_inference_steps = 10
    transform_floor_plan(input_image_path, output_image_path, num_inference_steps)
    assert Image.open(output_image_path) is not None, 'Output image not generated.'

# call_test_function_code --------------------

test_transform_floor_plan()