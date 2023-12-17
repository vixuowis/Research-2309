# requirements_file --------------------

!pip install -U opencv, diffusers, transformers, accelerate

# function_import --------------------

import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

# function_code --------------------

def generate_visual_representation(input_image_path):
    # Load room plan image
    image = load_image(input_image_path)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    # Use Canny edge detection
    image_canny = cv2.Canny(image, low_threshold, high_threshold)
    image_canny = image_canny[:, :, None]
    image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
    image_canny = Image.fromarray(image_canny)
    
    # Create ControlNetModel instance
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-canny', torch_dtype=torch.float16)
    # Create pipeline instance
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    
    # Generate the visual representation
    transformed_image = pipe('visual representation', image_canny, num_inference_steps=20).images[0]
    return transformed_image

# test_function_code --------------------

def test_generate_visual_representation():
    print("Testing generate_visual_representation function.")
    transform_result = generate_visual_representation('sample_room_plan.jpg')
    assert transform_result is not None, 'Function failed to return a result.'
    assert hasattr(transform_result, 'size'), 'Result is not an image.'
    print("Test passed.")