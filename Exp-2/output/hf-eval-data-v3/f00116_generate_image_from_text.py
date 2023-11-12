# function_import --------------------

import torch
from diffusers.utils import load_image
from PIL import Image
from controlnet_aux import HEDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_image_from_text(prompt: str, control_image_path: str, checkpoint: str = 'lllyasviel/control_v11p_sd15_scribble') -> None:
    """
    Generate an image from a text description using a pretrained ControlNet model.

    Args:
        prompt (str): The text description of the desired image.
        control_image_path (str): The path to the control image.
        checkpoint (str, optional): The model checkpoint to use. Defaults to 'lllyasviel/control_v11p_sd15_scribble'.

    Returns:
        None. The generated image is saved to 'images/image_out.png'.
    """
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(0)
    scribble_image = Image.open(control_image_path)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=scribble_image).images[0]
    image.save('images/image_out.png')

# test_function_code --------------------

def test_generate_image_from_text():
    """
    Test the generate_image_from_text function.
    """
    # Test with a simple prompt and control image
    generate_image_from_text('a simple house', './images/control.png')
    assert os.path.exists('images/image_out.png'), 'Image not generated'
    # Test with a complex prompt and control image
    generate_image_from_text('a complex scene with multiple objects', './images/control.png')
    assert os.path.exists('images/image_out.png'), 'Image not generated'
    # Test with a different model checkpoint
    generate_image_from_text('a simple house', './images/control.png', 'lllyasviel/control_v11p_sd15_scribble')
    assert os.path.exists('images/image_out.png'), 'Image not generated'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_generate_image_from_text())