# function_import --------------------

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from controlnet_aux import MLSDdetector
import torch
from PIL import Image

# function_code --------------------

def generate_image(prompt: str, num_inference_steps: int = 30, seed: int = 0, image_path: str = 'images/rendered_living_room.png') -> None:
    """
    Generate an image based on the given prompt using a pre-trained ControlNetModel.

    Args:
        prompt (str): The text prompt to generate the image from.
        num_inference_steps (int, optional): The number of inference steps to perform. Defaults to 30.
        seed (int, optional): The seed for the random number generator. Defaults to 0.
        image_path (str, optional): The path to save the generated image. Defaults to 'images/rendered_living_room.png'.

    Returns:
        None
    """
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_mlsd', torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16
    )

    control_image = processor(image)
    generated_image = pipe(prompt, num_inference_steps=num_inference_steps, generator=torch.manual_seed(seed), image=control_image).images[0]
    generated_image.save(image_path)

# test_function_code --------------------

def test_generate_image():
    """
    Test the generate_image function.

    Raises:
        AssertionError: If the function does not generate an image file at the specified path.
    """
    test_prompt = 'luxury living room with a fireplace'
    test_image_path = 'test_images/test_rendered_living_room.png'
    generate_image(test_prompt, image_path=test_image_path)
    assert Image.open(test_image_path) is not None, 'No image file generated at the specified path.'

# call_test_function_code --------------------

test_generate_image()