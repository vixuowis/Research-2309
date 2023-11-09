# function_import --------------------

from PIL import Image
import torch
from diffusers.utils import load_image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def add_building_and_river(image_path: str, output_path: str = 'image_out.png'):
    '''
    Add a building and a river to a given landscape image using a pre-trained ControlNetModel.

    Args:
        image_path (str): The path to the source landscape image.
        output_path (str, optional): The path to save the transformed image. Defaults to 'image_out.png'.

    Returns:
        None. The transformed image is saved to the specified output path.
    '''
    control_image = load_image(image_path).convert('RGB')
    prompt = 'add a building and a river'

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

    Returns:
        None. An assertion error is raised if the function does not work as expected.
    '''
    # Use a sample landscape image for testing
    image_path = 'test_landscape.jpg'
    output_path = 'test_image_out.png'

    # Call the function to be tested
    add_building_and_river(image_path, output_path)

    # Check if the output image is created
    assert os.path.exists(output_path), 'The output image does not exist.'

# call_test_function_code --------------------

test_add_building_and_river()