# requirements_file --------------------

!pip install -U torch diffusers

# function_import --------------------

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# function_code --------------------

def generate_fantasy_image(prompt: str, model_id: str, output_path: str) -> None:
    """
    Generates an image from a text prompt using a pre-trained model.

    Args:
        prompt (str): Text description of the image to generate.
        model_id (str): Pre-trained model ID from Hugging Face.
        output_path (str): Path to save the generated image.

    Returns:
        None

    Raises:
        RuntimeError: If there is an issue with model loading or image generation.
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    image_result = pipe(prompt, num_inference_steps=25).images[0]
    image_result.save(output_path)

# test_function_code --------------------

def test_generate_fantasy_image():
    print("Testing started.")
    # Create a prompt for the test
    test_prompt = "Test prompt: a peaceful scene in a lush green forest with a crystal-clear river"

    # Assume model_id is already known for testing purposes
    test_model_id = 'darkstorm2150/Protogen_v5.8_Official_Release'

    # Output path for the generated image
    test_output_path = "test_output.jpg"

    # Test case 1: Check if function runs without error
    print("Testing case [1/1] started.")
    try:
        generate_fantasy_image(test_prompt, test_model_id, test_output_path)
        print("Test case [1/1] passed.")
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_fantasy_image()