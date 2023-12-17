# requirements_file --------------------

!pip install -U diffusers torch

# function_import --------------------

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# function_code --------------------

def generate_magical_forest_image(prompt, control_net_checkpoint, stable_diffusion_checkpoint, seed=0, num_inference_steps=30):
    """
    Generates an image based on the provided text prompt using a ControlNet model and Stable Diffusion.

    Args:
        prompt (str): The text prompt describing the image to generate.
        control_net_checkpoint (str): The checkpoint path or identifier for the ControlNetModel.
        stable_diffusion_checkpoint (str): The checkpoint path or identifier for the StableDiffusionControlNetPipeline.
        seed (int, optional): Seed for random number generator to ensure reproducibility. Default to 0.
        num_inference_steps (int, optional): The number of inference steps for image generation. Default to 30.

    Returns:
        PIL.Image.Image: The generated image based on the text prompt.

    Raises:
        ValueError: If any of the checkpoints are not provided correctly.
    """
    if not prompt or not control_net_checkpoint or not stable_diffusion_checkpoint:
        raise ValueError('Prompt and checkpoint paths must be provided.')

    # Load the ControlNetModel with the specified torch dtype
    controlnet = ControlNetModel.from_pretrained(control_net_checkpoint, torch_dtype=torch.float16)

    # Load the StableDiffusionControlNetPipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        stable_diffusion_checkpoint, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Set the random seed for reproducibility
    generator = torch.manual_seed(seed)

    # Generate the image
    generated_image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]

    return generated_image

# test_function_code --------------------

def test_generate_magical_forest_image():
    print("Testing started.")

    # Test case computation is placeholder as actual implementation would require heavy computation
    # Replace 'image_not_none' assertion with actual image validation

    # Test case 1: Check if image is not None
    print("Testing case [1/1] started.")
    image = generate_magical_forest_image(
        "A magical forest with unicorns and a rainbow.",
        'lllyasviel/control_v11p_sd15_softedge',
        'runwayml/stable-diffusion-v1-5',
        seed=0,
        num_inference_steps=30
    )
    assert image is not None, f"Test case [1/1] failed: The image should not be None."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_magical_forest_image()