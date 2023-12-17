# requirements_file --------------------

import subprocess

requirements = ["diffusers", "controlnet_aux"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import torch
from diffusers.utils import load_image
from PIL import Image
from controlnet_aux import HEDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_image_from_scribble(prompt, scribble_image_path):
    """Generate an image from a text description using a scribble image as a control input.

    Args:
        prompt (str): Text description of the desired image.
        scribble_image_path (str): Path to the scribble control image.

    Returns:
        PIL.Image.Image: The generated image.

    Raises:
        ValueError: If the prompt or scribble_image_path is empty.
        FileNotFoundError: If the scribble_image_path does not exist.
    """
    if not prompt:
        raise ValueError('The prompt cannot be empty.')
    if not scribble_image_path or not os.path.exists(scribble_image_path):
        raise FileNotFoundError('The scribble image path does not exist or is invalid.')

    checkpoint = 'lllyasviel/control_v11p_sd15_scribble'
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(0)

    scribble_image = Image.open(scribble_image_path)
    control_image = pipe(prompt, num_inference_steps=30, generator=generator, image=scribble_image).images[0]

    return control_image

# test_function_code --------------------

def test_generate_image_from_scribble():
    print('Testing started.')
    # Assuming there's an existing function `load_dataset` to fetch test data
    dataset = load_dataset('test_dataset')
    sample_data = dataset[0]

    # Test case 1: Test with valid prompt and scribble image path
    print('Testing case [1/3] started.')
    generated_image = generate_image_from_scribble('a sunset behind mountains', './test_images/test_scribble.png')
    assert isinstance(generated_image, Image.Image), 'Test case [1/3] failed: The result should be an instance of PIL.Image.Image.'

    # Test case 2: Test with an empty prompt
    print('Testing case [2/3] started.')
    try:
        generate_image_from_scribble('', './test_images/test_scribble.png')
        assert False, 'Test case [2/3] failed: Function did not raise a ValueError for empty prompt.'
    except ValueError as e:
        assert str(e) == 'The prompt cannot be empty.', 'Test case [2/3] failed: The exception message is incorrect.'

    # Test case 3: Test with an invalid scribble image path
    print('Testing case [3/3] started.')
    try:
        generate_image_from_scribble('a sunset behind mountains', '')
        assert False, 'Test case [3/3] failed: Function did not raise a FileNotFoundError for invalid scribble image path.'
    except FileNotFoundError as e:
        assert str(e) == 'The scribble image path does not exist or is invalid.', 'Test case [3/3] failed: The exception message is incorrect.'

    print('Testing finished.')

# call_test_function_line --------------------

test_generate_image_from_scribble()