# requirements_file --------------------

!pip install -U diffusers controlnet_aux transformers

# function_import --------------------

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image

# function_code --------------------

def generate_enhanced_architecture_image(image_path):
    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    image = load_image(image_path)
    image = mlsd(image)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    generated_image = pipe(image, num_inference_steps=20).images[0]
    output_path = "images/generated_architecture.png"
    generated_image.save(output_path)
    print(f"Generated image saved at {output_path}")
    return output_path

# test_function_code --------------------

def test_generate_enhanced_architecture_image():
    print("Testing started.")
    image_path = "path_to_test_architectural_image.jpg"
    print("Testing case [1/1] started.")
    output_path = generate_enhanced_architecture_image(image_path)
    assert output_path.endswith('generated_architecture.png'), f"Test case [1/1] failed: The function did not return the expected output image path."
    print("Testing finished.")

test_generate_enhanced_architecture_image()