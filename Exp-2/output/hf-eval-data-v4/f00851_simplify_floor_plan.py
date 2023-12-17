# requirements_file --------------------

!pip install -U diffusers transformers accelerate controlnet_aux

# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def simplify_floor_plan(image_path: str, output_path: str) -> None:
    """
    Simplify the floor plan image into a straight line drawing.

    Parameters:
        image_path (str): The file path to the floor plan image.
        output_path (str): The file path where the simplified image should be saved.
    """
    # Load MLSD (Multi-Layered Single Shot Detector)
    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    
    # Load the floor plan image
    floor_plan_img = load_image(image_path)
    
    # Transform the image using MLSDdetector
    floor_plan_img = mlsd(floor_plan_img)
    
    # Load the ControlNet model with float16
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-mlsd', torch_dtype=torch.float16)
    
    # Create the pipeline with the ControlNet model
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    
    # Update the scheduler to UniPCMultistepScheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Enable memory efficient attention and CPU offloading for performance
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    
    # Process the floor plan image
    result_img = pipe(floor_plan_img, num_inference_steps=20).images[0]
    
    # Save the resulting image
    result_img.save(output_path)

# test_function_code --------------------

def test_simplify_floor_plan():
    print("Testing simplify_floor_plan function.")
    # Assume test floor plan and output paths are given
    test_image_path = 'test_floor_plan.png'
    test_output_path = 'test_floor_plan_simplified.png'

    # Function execution
    simplify_floor_plan(test_image_path, test_output_path)

    # Load the output image and check if file exists
    try:
        with Image.open(test_output_path) as img:
            assert img is not None, "Output image not found or invalid."
    except FileNotFoundError:
        assert False, "Output file not created."

    print("Testing complete. Function simplify_floor_plan passed.")