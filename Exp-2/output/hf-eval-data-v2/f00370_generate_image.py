# function_import --------------------

import torch
from huggingface_hub import HfApi
from diffusers.utils import load_image
from PIL import Image
from controlnet_aux import NormalBaeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_image(prompt: str, image_url: str, checkpoint: str = 'lllyasviel/control_v11p_sd15_normalbae'):
    '''
    Generate an image from a given input text using the ControlNetModel from Hugging Face.

    Args:
    prompt: str: The input text to generate the image from.
    image_url: str: The URL of the input image.
    checkpoint: str: The checkpoint for the pretrained model. Default is 'lllyasviel/control_v11p_sd15_normalbae'.

    Returns:
    The generated image.
    '''
    image = load_image(image_url)
    processor = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(image)

    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(33)
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    return generated_image

# test_function_code --------------------

def test_generate_image():
    '''
    Test the generate_image function.
    '''
    prompt = 'A head full of roses'
    image_url = 'https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/images/input.png'
    generated_image = generate_image(prompt, image_url)
    assert isinstance(generated_image, Image.Image), 'The output should be an instance of PIL.Image.Image.'

# call_test_function_code --------------------

test_generate_image()