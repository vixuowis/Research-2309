# function_import --------------------

from PIL import Image
import torch
from controlnet_aux import NormalBaeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_book_cover(input_image_path: str, output_image_path: str, prompt: str = 'A head full of roses') -> None:
    '''
    Generate a book cover based on a given prompt and input image.

    Args:
        input_image_path (str): The path to the input image file.
        output_image_path (str): The path to save the generated image.
        prompt (str, optional): The prompt for the image generation. Defaults to 'A head full of roses'.

    Returns:
        None
    '''
    image = Image.open(input_image_path)
    processor = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(image)

    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_normalbae', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(33)
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    generated_image.save(output_image_path)

# test_function_code --------------------

def test_generate_book_cover():
    '''
    Test the function generate_book_cover.
    '''
    import os
    input_image_path = 'https://placekitten.com/200/300'
    output_image_path = 'test_output.png'
    prompt = 'A cat full of roses'
    generate_book_cover(input_image_path, output_image_path, prompt)
    assert os.path.exists(output_image_path), 'Output image not found.'
    os.remove(output_image_path)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_book_cover()