# requirements_file --------------------

!pip install -U torch, PIL, controlnet_aux, diffusers

# function_import --------------------

import torch
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_image_with_pose(text_description, pose_image_path, output_image_path):
    """
    Generates an image with specified pose and position based on a textual description
    
    Parameters:
    - text_description (str): The text describing the scene and/or the appearance of the person
    - pose_image_path (str): The file path to an image containing the positions and poses of the objects
    - output_image_path (str): The file path where the generated image will be saved
    
    Returns:
    - The generated image saved to the specified output path
    """
    # Load the pretrained ControlNetModel checkpoint
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose', torch_dtype=torch.float16)
    
    # Create an OpenposeDetector using the pretrained model
    openpose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    
    # Load the pose image and generate the control image containing the positions and poses
    input_pose_image = Image.open(pose_image_path)
    control_image = openpose_detector(input_pose_image, hand_and_face=True)
    
    # Create the StableDiffusionControlNetPipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Enable CPU offload to avoid GPU memory overflow
    pipe.enable_model_cpu_offload()
    
    # Generate the image based on the text description and control image
    generator = torch.manual_seed(0)
    output_image = pipe(text_prompt=text_description, num_inference_steps=30, generator=generator, image=control_image).images[0]
    
    # Save the generated image to the output path
    output_image.save(output_image_path)
    
    return output_image

# test_function_code --------------------

from PIL import Image
import os

def test_generate_image_with_pose():
    print("Testing generate_image_with_pose function.")
    sample_text = "A professional basketball player dunking the ball"
    sample_pose_image_path = "sample_pose.jpg"  # Sample pose image path
    sample_output_image_path = "generated_image.jpg"  # Output path for the generated image
    
    # Test case 1: Check if the function completes without errors
    print("Testing case [1/2] started.")
    try:
        generate_image_with_pose(sample_text, sample_pose_image_path, sample_output_image_path)
        assert True, "Function executed without errors."
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"
    
    # Test case 2: Check if the image is saved to the specified output path
    print("Testing case [2/2] started.")
    assert os.path.exists(sample_output_image_path), f"Test case [2/2] failed: Output image was not saved at {sample_output_image_path}."
    
    print("Testing finished.")

# Run the test function
test_generate_image_with_pose()