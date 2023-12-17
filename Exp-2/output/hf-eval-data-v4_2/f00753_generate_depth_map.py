# requirements_file --------------------

!pip install -U diffusers transformers accelerate PIL numpy torch

# function_import --------------------

from transformers import pipeline
from diffusers import ControlNetModel
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image

# function_code --------------------

def generate_depth_map(image_path: str, output_path: str) -> Image.Image:
    """
    Generates a depth map from an input image of a street filled with people.

    Args:
        image_path (str): The file path to the street image with people.
        output_path (str): The file path where the depth map image will be saved.

    Returns:
        Image.Image: The PIL image object of the depth map.

    Raises:
        FileNotFoundError: If the input image file does not exist.
        ValueError: If there is an issue during the computation of the depth map.
    """
    # Load the model
    model = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-depth', torch_dtype=torch.float16)

    # Create the depth estimator pipeline
    depth_estimator = pipeline('depth-estimation', model=model)

    # Load the input image
    input_image = load_image(image_path)

    # Predict the depth
    depth_image = depth_estimator(input_image)['depth']

    # Process the depth to be saved as an image
    depth_image_array = np.array(depth_image)
    depth_image_array = depth_image_array[:, :, None] * np.ones(3, dtype=np.float32)[None, None, :]
    output_image = Image.fromarray(depth_image_array.astype(np.uint8))

    # Save the output
    output_image.save(output_path)

    return output_image

# test_function_code --------------------

def test_generate_depth_map():
    print("Testing started.")
    # Define the input and output paths for the test
    test_input_path = './test_image.png'
    test_output_path = './test_output.png'

    # Ensure the test image file exists
    assert os.path.exists(test_input_path), 'Test image file does not exist.'

    # Testing case
    print("Testing case [1/1] started.")
    try:
        result_image = generate_depth_map(test_input_path, test_output_path)
        assert isinstance(result_image, Image.Image), 'Output is not an image.'
        assert os.path.exists(test_output_path), 'Depth map output file not found.'
    except Exception as e:
        assert False, f'Test case failed with error: {e}'
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_depth_map()