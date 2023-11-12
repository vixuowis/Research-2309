# function_import --------------------

from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image
import os

# function_code --------------------

def estimate_depth(image_url: str, output_file: str) -> None:
    '''
    Estimate the depth of the entities in the given image and save the depth-estimated image.

    Args:
        image_url (str): The URL of the image to estimate depth.
        output_file (str): The file path to save the depth-estimated image.

    Returns:
        None
    '''
    depth_estimator = pipeline('depth-estimation')
    image = load_image(image_url)
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-depth', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    depth_output = pipe('Depth Estimation', image, num_inference_steps=20).images[0]
    depth_output.save(output_file)

# test_function_code --------------------

def test_estimate_depth():
    '''
    Test the function estimate_depth.
    '''
    image_url = 'https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png'
    output_file = './images/stormtrooper_depth_out.png'
    estimate_depth(image_url, output_file)
    assert os.path.exists(output_file), 'Test Failed: Output file not found.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_estimate_depth()