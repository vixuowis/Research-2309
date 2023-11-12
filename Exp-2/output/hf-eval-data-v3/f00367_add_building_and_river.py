# function_import --------------------

from PIL import Image
import torch
from diffusers.utils import load_image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def add_building_and_river(image_path: str, prompt: str, output_path: str = 'image_out.png'):
    '''
    Add a building and a river to a given landscape image using a pretrained ControlNetModel.

    Args:
        image_path (str): The path to the input landscape image.
        prompt (str): The transformation to apply to the image, in this case 'add a building and a river'.
        output_path (str, optional): The path to save the transformed image. Defaults to 'image_out.png'.

    Returns:
        None
    '''
    control_image = load_image(image_path).convert('RGB')

    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11e_sd15_ip2p', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(0)

    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    image.save(output_path)

# test_function_code --------------------

def test_add_building_and_river():
    '''
    Test the function add_building_and_river.
    '''
    # Test case 1: Add a building and a river to a landscape image
    add_building_and_river('landscape.jpg', 'add a building and a river', 'image_out1.png')
    assert Image.open('image_out1.png') is not None

    # Test case 2: Add a building and a river to another landscape image
    add_building_and_river('landscape2.jpg', 'add a building and a river', 'image_out2.png')
    assert Image.open('image_out2.png') is not None

    # Test case 3: Add a building and a river to a third landscape image
    add_building_and_river('landscape3.jpg', 'add a building and a river', 'image_out3.png')
    assert Image.open('image_out3.png') is not None

    return 'All Tests Passed'

# call_test_function_code --------------------

test_add_building_and_river()