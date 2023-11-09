from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector


def estimate_human_pose(image_path):
    '''
    This function estimates the human pose from an image.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    str: The path to the output image with the estimated pose.
    '''
    # Create an instance of the OpenposeDetector class
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    
    # Load the image data from a file
    image = Image.open(image_path)
    
    # Perform human pose estimation using the OpenposeDetector
    image = openpose(image)
    
    # Initialize the ControlNet model
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-openpose', torch_dtype=torch.float16)
    
    # Perform the pose estimation using the pipeline
    image = pipe('chef in the kitchen', image, num_inference_steps=20).images[0]
    
    # Save the output image
    output_path = 'images/pose_out.png'
    image.save(output_path)
    
    return output_path