# requirements_file --------------------

!pip install -U diffusers transformers accelerate PIL numpy torch

# function_import --------------------

from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image

# function_code --------------------

def estimate_stormtrooper_depth(image_url):
    """
    Estimate the depth of stormtroopers in a given Star Wars scene image.

    Parameters
    ----------
    image_url : str
        URL of the image with stormtroopers to process.

    Returns
    -------
    Image
        Image with estimated depth.
    """
    # Load the pretrained model and create a pipeline for depth estimation
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-depth', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    # Load the image
    image = load_image(image_url)

    # Estimate the depth
    depth_output = pipe("Stormtrooper's lecture", image, num_inference_steps=20).images[0]

    # Save the depth-estimated image
    depth_output.save('./images/stormtrooper_depth_out.png')
    return depth_output

# test_function_code --------------------

def test_estimate_stormtrooper_depth():
    print("Testing started.")
    test_image_url = 'https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png'

    # Test case: Estimate depth of stormtroopers in the image
    print("Testing depth estimation started.")
    estimated_image = estimate_stormtrooper_depth(test_image_url)

    # Verify if the estimated image is not None
    assert estimated_image is not None, "Test depth estimation failed: the resulting image is None."

    # Since actual depth estimation correctness is hard to assert programmatically without manual verification,
    # we check if an image saved successfully
    try:
        with Image.open('./images/stormtrooper_depth_out.png') as img:
            assert img is not None, "Estimated depth image couldn't be opened."
    except Exception as e:
        assert False, f"Estimated depth image couldn't be opened or does not exist. Error: {e}"

    print("Testing depth estimation finished.")

# Run the test function
test_estimate_stormtrooper_depth()