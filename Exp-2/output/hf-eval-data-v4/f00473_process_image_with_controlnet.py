# requirements_file --------------------

!pip install -U diffusers transformers accelerate controlnet_aux

# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def process_image_with_controlnet(input_image_path, output_image_path):
    # Load the pretrained M-LSD model for detecting straight lines
    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    image = load_image(input_image_path)
    image = mlsd(image)

    # Load the ControlNet model
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Enhance memory efficiency
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    # Process the image through the pipeline
    processed_image = pipe(image, num_inference_steps=20).images[0]
    processed_image.save(output_image_path)
    return output_image_path

# test_function_code --------------------

def test_process_image_with_controlnet():
    print("Testing process_image_with_controlnet function.")
    output_path = process_image_with_controlnet('input_image.png', 'output_image.png')
    assert os.path.exists(output_path), f"Output image not found at {output_path}"
    print("Test passed: Output image found.")

# Running the test
if __name__ == '__main__':
    test_process_image_with_controlnet()