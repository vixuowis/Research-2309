# function_import --------------------

from PIL import Image
import torch
from controlnet_aux import NormalBaeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_book_cover(input_image_path: str, output_image_path: str, prompt: str = 'A head full of roses'):
    """
    Generate a book cover image based on a given prompt using a pretrained ControlNetModel.

    Args:
        input_image_path (str): The path to the input image file.
        output_image_path (str): The path to save the generated image file.
        prompt (str, optional): The prompt describing the image to generate. Defaults to 'A head full of roses'.

    Returns:
        None
    """
    image = Image.open(input_image_path)
    processor = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(image)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_normalbae', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(33)
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    generated_image.save(output_image_path)

# test_function_code --------------------

def test_generate_book_cover():
    """
    Test the generate_book_cover function.
    """
    input_image_path = 'test_input_image.png'
    output_image_path = 'test_output_image.png'
    prompt = 'A head full of roses'
    generate_book_cover(input_image_path, output_image_path, prompt)
    assert os.path.exists(output_image_path), 'Output image not found.'

# call_test_function_code --------------------

test_generate_book_cover()