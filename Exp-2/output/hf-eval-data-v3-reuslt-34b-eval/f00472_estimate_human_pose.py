# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
import os

# function_code --------------------

def estimate_human_pose(image_path: str, output_path: str) -> None:
    """
    Estimate the human pose from an image of a user performing an exercise.

    Args:
        image_path (str): The path to the image file.
        output_path (str): The path to save the output image with estimated pose.

    Returns:
        None
    """    
    
    # create a detector object
    det = OpenposeDetector()
    
    # load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ControlNetModel(numJoints=det.numJoints, numParts=det.pairs, numCoords=2)
    pipeline = StableDiffusionControlNetPipeline(model=net, device=device)
    
    # create a scheduler object
    sched = UniPCMultistepScheduler(pipeline, diffusion_steps=1000, noise_schedule="cosine")
    
    # load the image to be processed
    image = Image.open(image_path)
        
    # estimate the human pose
    prediction = sched(image)
     
    # draw estimated human pose onto original image (in red)
    det.plotPoints(prediction, image=image) 
    
    # save output image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


# test_function_code --------------------

def test_estimate_human_pose():
    """
    Test the function estimate_human_pose.
    """
    # Test case 1
    estimate_human_pose('test_images/exercise1.jpg', 'test_output/pose1_out.png')
    assert os.path.exists('test_output/pose1_out.png')

    # Test case 2
    estimate_human_pose('test_images/exercise2.jpg', 'test_output/pose2_out.png')
    assert os.path.exists('test_output/pose2_out.png')

    # Test case 3
    estimate_human_pose('test_images/exercise3.jpg', 'test_output/pose3_out.png')
    assert os.path.exists('test_output/pose3_out.png')

    return 'All Tests Passed'


# call_test_function_code --------------------

test_estimate_human_pose()