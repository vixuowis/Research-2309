# requirements_file --------------------

!pip install -U torch diffusers

# function_import --------------------

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# function_code --------------------

def generate_fantasy_landscape_image(model_id, prompt, output_file_path):
    """
    Generate an image of a fantasy landscape using the provided model and text description.

    Parameters:
        model_id (str): ID of the text-to-image model to use.
        prompt (str): Text description of the image to generate.
        output_file_path (str): Path to save the generated image.

    Returns:
        None
    """
    # Load the model with the specified configuration
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    # Set the scheduler for the model
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # Move the model to GPU for faster processing
    pipe = pipe.to('cuda')

    # Generate the image
    image_result = pipe(prompt, num_inference_steps=25).images[0]
    # Save the image
    image_result.save(output_file_path)

# test_function_code --------------------

def test_generate_fantasy_landscape_image():
    print("Testing started.")

    # Test case 1: Check if function generates a file
    print("Testing case [1/1] started.")
    output_file_path = './test_result.jpg'
    prompt = 'a peaceful scene in a lush green forest with a crystal-clear river flowing through it, under a blue sky with fluffy white clouds'
    model_id = 'darkstorm2150/Protogen_v5.8_Official_Release'
    generate_fantasy_landscape_image(model_id, prompt, output_file_path)
    assert os.path.isfile(output_file_path), f"Test case [1/1] failed: No file generated at {output_file_path}"
    print("Testing finished.")

# Run the test function
test_generate_fantasy_landscape_image()