# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_cat_character_image(model_id='google/ddpm-ema-cat-256', output_file='cat_character_image.png'):
    """
    Generate a cartoon cat character image using a pre-trained Denoising Diffusion Probabilistic Model.

    Args:
        model_id (str): The ID of the pre-trained diffusers model to use for image generation.
        output_file (str): The file path where the generated image will be saved.

    Returns:
        None: The generated image is saved to the specified output file.
    """
    # Load the pre-trained DDPM model
    ddpm = DDPMPipeline.from_pretrained(model_id)
    
    # Generate an image
    image = ddpm().images[0]
    
    # Save the generated image
    image.save(output_file)

# test_function_code --------------------

def test_generate_cat_character_image():
    print("Testing started.")

    # Test case: Generating and saving an image.
    generate_cat_character_image()  # Generate and save image with default parameters
    print("Testing case 1: Image generated and saved.")
    assert os.path.exists('cat_character_image.png'), "Test case failed: Image file 'cat_character_image.png' does not exist."
    print("Testing finished.")

# Run the test function
test_generate_cat_character_image()