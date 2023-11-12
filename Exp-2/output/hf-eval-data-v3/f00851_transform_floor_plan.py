# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def transform_floor_plan(input_image_path: str, output_image_path: str, num_inference_steps: int = 20):
    '''
    Transforms the floor plan images into simple straight line drawings.

    Args:
        input_image_path (str): The path to the input floor plan image.
        output_image_path (str): The path to save the output transformed image.
        num_inference_steps (int, optional): The number of inference steps to process the image. Defaults to 20.

    Returns:
        None
    '''
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
    '''
    Tests the transform_floor_plan function.
    '''
    transform_floor_plan('test_images/floor_plan1.png', 'test_images/floor_plan1_simplified.png')
    assert Image.open('test_images/floor_plan1_simplified.png') is not None

    transform_floor_plan('test_images/floor_plan2.png', 'test_images/floor_plan2_simplified.png')
    assert Image.open('test_images/floor_plan2_simplified.png') is not None

    transform_floor_plan('test_images/floor_plan3.png', 'test_images/floor_plan3_simplified.png')
    assert Image.open('test_images/floor_plan3_simplified.png') is not None

    return 'All Tests Passed'

# call_test_function_code --------------------

test_transform_floor_plan()