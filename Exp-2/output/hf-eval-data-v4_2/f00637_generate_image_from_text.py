# requirements_file --------------------

pip install diffusers transformers accelerate scipy safetensors

# function_import --------------------

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# function_code --------------------

def generate_image_from_text(prompt: str, output_path: str) -> None:
    """
    Generates an image from the provided text using the Stable Diffusion model.
    
    Args:
        prompt (str): The text prompt to guide the image generation.
        output_path (str): The file path to save the generated image.
    
    Returns:
        None. The generated image is saved to the specified path.
    
    Raises:
        RuntimeError: If the model or scheduler can't be loaded correctly.
        IOError: If the image cannot be saved to the specified path.
    """
    model_id = 'stabilityai/stable-diffusion-2-1-base'
    # Load the scheduler with its subfolder
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder='scheduler')
    if not scheduler:
        raise RuntimeError('Failed to load the scheduler for image generation.')

    # Load the model with the specified scheduler
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    if not pipe:
        raise RuntimeError('Failed to load the Stable Diffusion pipeline for image generation.')

    # Set the device to run the model (choose 'cuda' or 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = pipe.to(device)

    # Generate the image
    image = pipe(prompt).images[0]

    # Save the image
    try:
        image.save(output_path)
    except IOError as e:
        raise IOError(f'Failed to save the image to {output_path}. Error: {e}')

# test_function_code --------------------

def test_generate_image_from_text():
    print("Testing started.")
    # No actual dataset loading or sample data is required as we are not testing the internal model functionality,
    # just our wrapping code logic.
  
    # Test case 1: Check function runs without error with a valid prompt and output path
    print("Testing case [1/3] started.")
    try:
        generate_image_from_text('An elegant swan on a moonlit lake', 'test_output1.png')
        print("Test case [1/3] passed.")
    except Exception as e:
        assert False, f"Test case [1/3] failed: {e}"
  
    # Test case 2: Check error is raised for invalid model or scheduler load
    print("Testing case [2/3] started.")
    try:
        generate_image_from_text('prompt', 'invalid_model_or_scheduler.png')
        assert False, "Test case [2/3] should have failed due to invalid model or scheduler."
    except RuntimeError as e:
        print("Test case [2/3] passed with RuntimeError as expected.")
  
    # Test case 3: Check error is raised for invalid output path
    print("Testing case [3/3] started.")
    try:
        generate_image_from_text('An abstract concept of time', '/invalid/path/output.png')
        assert False, "Test case [3/3] should have failed due to invalid output path."
    except IOError as e:
        print("Test case [3/3] passed with IOError as expected.")

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_image_from_text()