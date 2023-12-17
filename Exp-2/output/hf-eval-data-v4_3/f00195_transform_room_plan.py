# requirements_file --------------------

import subprocess

requirements = ["opencv-contrib-python", "diffusers", "transformers", "accelerate"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

# function_code --------------------

def transform_room_plan(input_image_path: str, output_image_path: str) -> None:
    """
    Transforms a room plan to a better visual representation using a pre-trained ControlNet model.

    Args:
        input_image_path (str): The file path to the input room plan image.
        output_image_path (str): The file path where the transformed image will be saved.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input image file does not exist.
        Exception: If the transformation process fails.
    """
    # Load room plan image
    image = load_image(input_image_path)
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
    transformed_image = pipe('room_plan', image, num_inference_steps=20).images[0]
    transformed_image.save(output_image_path)

# test_function_code --------------------

import os

from PIL import Image

def test_transform_room_plan():
    print("Testing started.")
    input_image_path = 'room_plan_sample.jpg'
    output_image_path = 'room_plan_transformed_sample.png'

    # Test case 1: Input image file does not exist
    print("Testing case [1/3] started.")
    non_existing_image_path = 'non_existent.jpg'
    try:
        transform_room_plan(non_existing_image_path, output_image_path)
        raise AssertionError("Test case [1/3] failed: FileNotFoundError was not raised.")
    except FileNotFoundError:
        pass

    # Test case 2: Transformation process success
    print("Testing case [2/3] started.")
    # Assuming a valid input image path for testing
    transform_room_plan(input_image_path, output_image_path)
    assert os.path.isfile(output_image_path), f"Test case [2/3] failed: Output image not saved to {output_image_path}."

    # Test case 3: Output image validation
    print("Testing case [3/3] started.")
    with Image.open(output_image_path) as img:
        assert img.format == 'PNG', "Test case [3/3] failed: Output image is not in PNG format."
        assert img.size[0] > 0 and img.size[1] > 0, "Test case [3/3] failed: Output image dimensions are invalid."
    print("Testing finished.")

# call_test_function_line --------------------

test_transform_room_plan()