# requirements_file --------------------

!pip install -U pillow torch diffusers controlnet_aux

# function_import --------------------

from PIL import Image
import torch
from controlnet_aux import NormalBaeDetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_book_cover(input_image_path: str, prompt: str, seed: int = 33) -> Image:
    """
    Generates a book cover image based on the input image and text prompt.

    Args:
        input_image_path (str): Path to the input image.
        prompt (str): Prompt describing the desired output.
        seed (int): Seed for reproducibility. Default is 33.

    Returns:
        PIL.Image: The generated book cover image.

    Raises:
        FileNotFoundError: If the input image path does not exist.
        RuntimeError: If the model or pipeline fails to generate the image.
    """
    image = Image.open(input_image_path)
    processor = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(image)

    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_normalbae', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(seed)
    generated_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

    return generated_image

# test_function_code --------------------

def test_generate_book_cover():
    print("Testing started.")

    # Test case 1: Valid input image and prompt
    print("Testing case [1/1] started.")
    try:
        output_image = generate_book_cover('input_image.png', 'A head full of roses')
        assert isinstance(output_image, Image.Image), "Output is not an image"
    except Exception as e:
        assert False, f"Test case [1/1] failed with error: {e}"
    finally:
        print("Testing finished.")

# call_test_function_line --------------------

test_generate_book_cover()