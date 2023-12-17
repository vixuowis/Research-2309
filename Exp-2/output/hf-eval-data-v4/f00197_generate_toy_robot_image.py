# requirements_file --------------------

!pip install -U torch, diffusers, Pillow, controlnet_aux

# function_import --------------------

import torch
from controlnet_aux import MLSDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image
from diffusers.utils import load_image

# function_code --------------------

def generate_toy_robot_image(prompt, initial_image_path=None):
    # Load the MLSDdetector model
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    # Load an initial image or use a placeholder
    initial_image = load_image(initial_image_path) if initial_image_path else Image.new('RGB', (512, 512), 'white')

    # Create a control image
    control_image = processor(initial_image)

    # Load the ControlNet model
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_mlsd', torch_dtype=torch.float16)

    # Create a StableDiffusionControlNetPipeline instance
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)

    # Set the scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Option to enable CPU offloading
    pipe.enable_model_cpu_offload()

    # Define the random seed for image generation
    generator = torch.manual_seed(0)

    # Generate the controlled image
    result_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

    # Return the PIL image
    return result_image

# test_function_code --------------------

def test_generate_toy_robot_image():
    print("Testing generate_toy_robot_image() function.")

    # Test case 1: Without initial image
    print("Testing with no initial image.")
    result = generate_toy_robot_image('toy robot')
    assert result is not None, 'Image generation failed without initial image.'

    # Test case 2: With an initial image
    print("Testing with an initial image path.")
    sample_image_path = 'sample_image.png'  # Replace with the path to a valid initial image
    result_with_initial = generate_toy_robot_image('toy robot', sample_image_path)
    assert result_with_initial is not None, 'Image generation failed with initial image.'

    print("Test completed successfully.")

# Run the test function
test_generate_toy_robot_image()