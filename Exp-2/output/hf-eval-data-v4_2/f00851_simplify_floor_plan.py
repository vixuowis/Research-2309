# requirements_file --------------------

!pip install -U PIL diffusers torch controlnet_aux

# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def simplify_floor_plan(image_path: str, output_path: str) -> None:
    """
    Simplifies the floor plan image to a straight line drawing.

    Args:
        image_path (str): The file path for the input floor plan image.
        output_path (str): The file path where the simplified image will be saved.

    Returns:
        None: The function saves the simplified image to the given output_path.

    Raises:
        FileNotFoundError: If the input image_path does not exist.
        IOError: If there is an issue while saving the image to output_path.
    """
    # Ensure the prerequisites are met
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image file not found: {image_path}")

    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    floor_plan_img = load_image(image_path)
    floor_plan_img = mlsd(floor_plan_img)

    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    result_img = pipe(floor_plan_img, num_inference_steps=20).images[0]
    result_img.save(output_path)

# test_function_code --------------------

def test_simplify_floor_plan():
    print("Testing started.")
    # Create a test image_path and output_path
    image_path = 'test_floor_plan.png'
    output_path = 'test_floor_plan_simplified.png'

    # Testing case 1: Check if function runs without errors with valid inputs
    print("Testing case [1/1] started.")
    try:
        simplify_floor_plan(image_path, output_path)
        assert os.path.exists(output_path), f"Test case [1/1] failed: Expected output file was not created at {output_path}"
        print("Test case [1/1] passed.")
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")

    print("Testing finished.")

# call_test_function_line --------------------

test_simplify_floor_plan()