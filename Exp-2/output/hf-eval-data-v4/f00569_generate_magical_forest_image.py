# requirements_file --------------------

!pip install -U diffusers torch

# function_import --------------------

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_magical_forest_image(prompt):
    # Load the pre-trained ControlNetModel
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
    
    # Initialize the StableDiffusionControlNetPipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    # Set seed for reproducibility
    generator = torch.manual_seed(0)
    
    # Generate the image based on the given prompt
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    
    # Save the generated image to a file
    generated_image.save('generated_magical_forest_image.png')
    
    return 'generated_magical_forest_image.png'

# test_function_code --------------------

def test_generate_magical_forest_image():
    print("Testing started.")

    # Test case 1: Check if the function returns the correct file name
    print("Testing case [1/1] started.")
    expected_output = 'generated_magical_forest_image.png'
    actual_output = generate_magical_forest_image("A magical forest with unicorns and a rainbow.")
    assert actual_output == expected_output, f"Test case [1/1] failed: Expected {expected_output}, but got {actual_output}"
    print("Testing finished.")

# Run the test function
test_generate_magical_forest_image()