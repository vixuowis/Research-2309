# requirements_file --------------------

!pip install -U diffusers torch controlnet_aux

# function_import --------------------

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
from controlnet_aux import HEDdetector

# function_code --------------------

def generate_image_from_text_with_control(prompt, control_image_path, output_image_path):
    '''
    Generates an image from a text description, using a scribble image as a control input.

    Parameters:
        prompt (str): The text description of the desired image.
        control_image_path (str): The file path to the control scribble image.
        output_image_path (str): The file path where the generated image will be saved.

    Returns:
        None
    '''
    checkpoint = 'lllyasviel/control_v11p_sd15_scribble'
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)
    scribble_image = Image.open(control_image_path)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=scribble_image).images[0]
    image.save(output_image_path)

# test_function_code --------------------

def test_generate_image_from_text_with_control():
    print("Testing generate_image_from_text_with_control function.")

    # Test case 1: Check if the function executes without error for valid inputs
    try:
        generate_image_from_text_with_control('A sunset behind the mountains', 'control_img.png', 'output_img.png')
        print("Test case passed: The function executed without error.")
    except Exception as e:
        print("Test case failed: The function raised an exception.", e)

    # Additional test cases would include checking the existence of the output file,
    # the validity of the inputs, and the properties of the returned/generated image.

    print("Testing finished.")