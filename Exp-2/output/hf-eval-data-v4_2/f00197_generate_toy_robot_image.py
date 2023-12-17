# requirements_file --------------------

!pip install -U torch controlnet_aux diffusers

# function_import --------------------

import torch
from controlnet_aux import MLSDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_toy_robot_image(prompt="toy robot", initial_image_path=None):
    """
    Generates an image of a toy robot based on a text prompt using a pretrained ControlNet model.

    Args:
        prompt (str): The text prompt describing the toy robot.
        initial_image_path (str): Optional path to an initial image to be transformed.

    Returns:
        Image: The generated toy robot image.

    Raises:
        ValueError: If the initial_image_path is not None and the file does not exist.
    """
    if initial_image_path:
        if not os.path.exists(initial_image_path):
            raise ValueError(f"Initial image path does not exist: {initial_image_path}")
        initial_image = load_image(initial_image_path)
    else:
        initial_image = None

    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    control_image = processor(initial_image)
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_mlsd', torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
    return image

# test_function_code --------------------

def test_generate_toy_robot_image():
    print("Testing started.")
    # No real dataset to load, skipping dataset loading

    # Test case 1: Testing with default prompt
    print("Testing case [1/3] started.")
    result_image = generate_toy_robot_image()
    assert result_image, "Test case [1/3] failed: No image was generated."

    # Test case 2: Testing with custom prompt
    print("Testing case [2/3] started.")
    custom_prompt = "futuristic toy robot with lasers"
    result_image = generate_toy_robot_image(prompt=custom_prompt)
    assert result_image, "Test case [2/3] failed: No image was generated with custom prompt."

    # Test case 3: Testing with invalid initial_image path
    print("Testing case [3/3] started.")
    try:
        generate_toy_robot_image(initial_image_path="invalid_path.png")
        assert False, "Test case [3/3] failed: Exception for invalid initial_image path was not raised."
    except ValueError as e:
        assert str(e) == "Initial image path does not exist: invalid_path.png", "Test case [3/3] failed: Incorrect ValueError message."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_toy_robot_image()