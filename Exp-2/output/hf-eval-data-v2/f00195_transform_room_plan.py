# function_import --------------------

import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

# function_code --------------------

def transform_room_plan(image_path: str, output_path: str = 'room_plan_transformed.png', num_inference_steps: int = 20):
    """
    Transforms a room plan image to a better visual representation using a pre-trained ControlNet model.

    Args:
        image_path (str): Path to the input room plan image.
        output_path (str, optional): Path to save the transformed image. Defaults to 'room_plan_transformed.png'.
        num_inference_steps (int, optional): Number of inference steps to perform. Defaults to 20.

    Returns:
        None. The transformed image is saved to the specified output path.
    """
    # Load room plan image
    image = load_image(image_path)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
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
    Tests the transform_room_plan function by transforming a sample room plan image and checking the output.
    """
    # Define input and output paths
    input_path = 'test_room_plan.jpg'
    output_path = 'test_room_plan_transformed.png'

    # Call the function
    transform_room_plan(input_path, output_path)

    # Check the output
    assert os.path.exists(output_path), 'Output image not found.'
    output_image = Image.open(output_path)
    assert output_image is not None, 'Output image could not be opened.'

# call_test_function_code --------------------

test_transform_room_plan()