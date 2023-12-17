# requirements_file --------------------

import subprocess

requirements = ["diffusers", "transformers", "accelerate", "opencv-python"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from PIL import Image
from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import numpy as np
import cv2

# function_code --------------------

def generate_normal_map_from_image(image_path, output_path):
    """
    Generate a normal map from an object image using a pretrained ControlNetModel.

    Args:
        image_path (str): The file path to the input image.
        output_path (str): The file path where the output normal map image will be saved.

    Returns:
        str: The file path to the generated normal map image.

    Raises:
        FileNotFoundError: If the input image file does not exist.
    """
    image = load_image(image_path).convert('RGB')
    depth_estimator = pipeline('depth-estimation', model='Intel/dpt-hybrid-midas')
    image_depth = depth_estimator(image)['predicted_depth'][0].numpy()

    # Preprocess the depth image
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)
    bg_threshold = 0.4
    x = cv2.Sobel(image_depth, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threshold] = 0
    y = cv2.Sobel(image_depth, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threshold] = 0
    z = np.ones_like(x) * np.pi * 2.0
    image_normal = np.stack([x, y, z], axis=2)
    image_normal /= np.sqrt(np.sum(image_normal**2, axis=2, keepdims=True))
    image_normal = (image_normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-normal', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    
    image_normal_map = Image.fromarray(image_normal)
    image_normal_map.save(output_path)
    
    return output_path

# test_function_code --------------------

def test_generate_normal_map_from_image():
    print("Testing started.")
    # Test case 1: Valid image path
    print("Testing case [1/2] started.")
    output_path = generate_normal_map_from_image('valid_image_path.png', 'output_normal_map.png')
    assert os.path.exists(output_path), f"Test case [1/2] failed: Output file {output_path} does not exist."

    # Test case 2: Invalid image path
    try:
        print("Testing case [2/2] started.")
        generate_normal_map_from_image('invalid_image_path.png', 'output_normal_map.png')
        assert False, "Test case [2/2] failed: No exception raised for invalid image path."
    except FileNotFoundError as e:
        assert str(e) == "No such file or directory: 'invalid_image_path.png'", f"Test case [2/2] failed with unexpected exception message: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_normal_map_from_image()