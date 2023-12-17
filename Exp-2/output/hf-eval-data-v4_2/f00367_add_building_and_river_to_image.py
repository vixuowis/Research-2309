# requirements_file --------------------

!pip install -U diffusers transformers accelerate

# function_import --------------------

import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image

# function_code --------------------

def add_building_and_river_to_image(input_image_path: str, output_image_path: str) -> None:
    """
    Add a building and a river to the landscape image.

    Args:
        input_image_path (str): The file path to the input image.
        output_image_path (str): The file path where the output image will be saved.

    Returns:
        None: The function saves the modified image to the specified output path.

    Raises:
        FileNotFoundError: If the input_image_path does not correspond to a file.
    """
    # Load the input image
    control_image = load_image(input_image_path).convert('RGB')

    # Define the transformation prompt
    prompt = "add a building and a river"

    # Load the pre-trained ControlNet model
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11e_sd15_ip2p', torch_dtype=torch.float16)

    # Create the StableDiffusionControlNetPipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Set a fixed random seed for reproducibility
    generator = torch.manual_seed(0)

    # Apply the transformation and save the output image
    image_out = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image_out.save(output_image_path)

# test_function_code --------------------

def test_add_building_and_river_to_image():
    print("Testing started.")
    
    # Test case 1: Check if calling function with a valid input image saves a file
    print("Testing case [1/1] started.")
    input_image_path = 'test_landscape.jpg'
    output_image_path = 'test_image_out.png'

    # Call the function with the test image
    add_building_and_river_to_image(input_image_path, output_image_path)

    # Validate that the output file was saved
    assert os.path.isfile(output_image_path), f"Test case [1/1] failed: Output file '{output_image_path}' was not saved."

    print("Testing finished.")

# call_test_function_line --------------------

test_add_building_and_river_to_image()