# requirements_file --------------------

!pip install -U Pillow torch numpy cv2 transformers diffusers

# function_import --------------------

from PIL import Image
from transformers import pipeline
import numpy as np
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image

# function_code --------------------

def generate_normal_map_from_image(image_path):
    image = load_image(image_path).convert('RGB')
    depth_estimator = pipeline('depth-estimation', model='Intel/dpt-hybrid-midas')
    image_depth = depth_estimator(image)['predicted_depth'][0].numpy()

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
    image_normal = np.clip(image_normal * 127.5 + 127.5, 0, 255).astype(np.uint8)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-normal', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    image_normal_map = Image.fromarray(image_normal)
    return image_normal_map

# test_function_code --------------------

def test_generate_normal_map_from_image():
    print("Testing generate_normal_map_from_image function.")
    image_path = 'sample_image.png'  # A sample image path
    image_normal_map = generate_normal_map_from_image(image_path)
    assert isinstance(image_normal_map, Image.Image), "The function should return a PIL Image object."
    assert image_normal_map.mode == 'RGB', "The normal map image must be in RGB mode."
    print("All tests passed!")

test_generate_normal_map_from_image()