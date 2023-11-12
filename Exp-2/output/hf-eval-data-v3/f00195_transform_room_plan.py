# function_import --------------------

import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
import os

# function_code --------------------

def transform_room_plan(image_path: str, output_path: str = 'room_plan_transformed.png', low_threshold: int = 100, high_threshold: int = 200, num_inference_steps: int = 20):
    """
    Transforms a room plan image to a better visual representation using a pre-trained ControlNetModel.

    Args:
        image_path (str): Path to the input room plan image.
        output_path (str, optional): Path to save the transformed image. Defaults to 'room_plan_transformed.png'.
        low_threshold (int, optional): Lower threshold for Canny edge detection. Defaults to 100.
        high_threshold (int, optional): Higher threshold for Canny edge detection. Defaults to 200.
        num_inference_steps (int, optional): Number of inference steps for the pipeline. Defaults to 20.

    Returns:
        None
    """
    # Load room plan image
    image = load_image(image_path)
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    # Create ControlNetModel and pipeline
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-canny', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)

    # Process and save output
    transformed_image = pipe('room_plan', image, num_inference_steps=num_inference_steps).images[0]
    transformed_image.save(output_path)

# test_function_code --------------------

def test_transform_room_plan():
    """
    Tests the transform_room_plan function.
    """
    # Test with default parameters
    transform_room_plan('room_plan.jpg')
    assert os.path.exists('room_plan_transformed.png'), 'Test failed: Output image not found.'

    # Test with custom parameters
    transform_room_plan('room_plan.jpg', 'custom_output.png', 50, 150, 10)
    assert os.path.exists('custom_output.png'), 'Test failed: Custom output image not found.'

    return 'All tests passed.'

# call_test_function_code --------------------

test_transform_room_plan()