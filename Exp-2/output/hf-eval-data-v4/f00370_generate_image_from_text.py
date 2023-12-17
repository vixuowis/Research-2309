# requirements_file --------------------

!pip install -U diffusers transformers accelerate controlnet_aux

# function_import --------------------

import torch
from huggingface_hub import HfApi
from diffusers.utils import load_image
from PIL import Image
from controlnet_aux import NormalBaeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_image_from_text(prompt, input_image_url):
    # Load the provided image from the URL
    image = load_image(input_image_url)

    # Pre-process the input image
    processor = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(image)

    # Load the ControlNet model with the given checkpoint
    checkpoint = 'lllyasviel/control_v11p_sd15_normalbae'
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)

    # Create a pipeline with the ControlNet model
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Set the seed for reproducibility
    generator = torch.manual_seed(33)

    # Generate an image with the given prompt and the pre-processed image
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

    # Return the generated image
    return generated_image

# test_function_code --------------------

def test_generate_image_from_text():
    print("Testing generate_image_from_text function.")

    # Test case: generate image from given text and image URL
    prompt = "A head full of roses"
    input_image_url = 'https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/images/input.png'
    generated_image = generate_image_from_text(prompt, input_image_url)

    # Check if an image was returned
    assert isinstance(generated_image, Image.Image), "The function should return an instance of PIL.Image.Image."

    print("All tests for generate_image_from_text passed.")