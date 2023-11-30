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
    detector = OpenposeDetector(
        weights_dir='./weights/openpose/',
        model_name="COCO")
    detections, image_rgb = detector.detect_humans(image_path)

    ndet = len(detections["body"][0])  # number of people detected

    for i in range(ndet):
        pose_keypoints = detections["body"][i].tolist()
        body_keypoints = torch.tensor([pose_keypoints]).float().to('cuda')
        output_image = Image.fromarray(((image_rgb * 255 / image_rgb.max())[:, :, [2,1,0]]).astype(np.uint8))
        draw_body(output_image, body_keypoints)
        # save image
        output_image.save(os.path.join(output_path,'{}_pose.png'.format(i)))


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