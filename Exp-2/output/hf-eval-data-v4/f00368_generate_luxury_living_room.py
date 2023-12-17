# requirements_file --------------------

!pip install -U diffusers transformers accelerate controlnet_aux

# function_import --------------------

import torch
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
from controlnet_aux import MLSDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

# function_code --------------------

def generate_luxury_living_room(control_img_url, prompt='luxury living room with a fireplace', save_path='images/rendered_living_room.png'):
    # Load the image from URL
    image = load_image(control_img_url)

    # Initialize MLSDdetector and ControlNet model
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_mlsd', torch_dtype=torch.float16)

    # Create the StableDiffusionControlNetPipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )

    # Prepare control image and generate the image
    control_image = processor(image)
    generated_image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0), image=control_image).images[0]

    # Save generated image
    generated_image.save(save_path)
    return save_path

# test_function_code --------------------

from diffusers.utils import load_image

# Test function for generate_luxury_living_room
def test_generate_luxury_living_room():
    print("Testing generate_luxury_living_room function.")
    control_image_url = 'https://example.com/control_image.jpg'
    output_path = 'test_images/test_rendered_living_room.png'

    # Test with a sample control image URL
    result_path = generate_luxury_living_room(control_img_url=control_image_url, save_path=output_path)
    assert Path(result_path).is_file(), f"Failed to generate image: {result_path} does not exist."
    print("Test passed: Luxury living room image generated successfully.")

# Run the test
test_generate_luxury_living_room()