# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_insect_image(model_name: str) -> str:
    """
    Generates an image of an insect using the specified diffusion model.

    Args:
        model_name: The name of the pretrained diffusion model to use.

    Returns:
        The filename of the saved insect image.

    Raises:
        ValueError: If the model_name is empty or None.
    """
    if not model_name:
        raise ValueError('Model name must be provided.')
    
    # Load the diffusion model
    pipeline = DDPMPipeline.from_pretrained(model_name)
    
    # Generate the image
    generated_image = pipeline().images[0]
    
    # Save the image
    file_name = 'insect_image.png'
    generated_image.save(file_name)
    
    return file_name

# test_function_code --------------------

def test_generate_insect_image():
    print("Testing started.")
    model_name = 'schdoel/sd-class-AFHQ-32'  # Pretrained model name for the test
    
    print("Testing case [1/1] started.")
    file_name = generate_insect_image(model_name)
    assert os.path.exists(file_name), f"Test case failed: Image file '{file_name}' does not exist."
    
    # Clean up the generated file after test
    os.remove(file_name)
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_insect_image()