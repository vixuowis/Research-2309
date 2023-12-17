# requirements_file --------------------

!pip install -U diffusers Pillow

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_bedroom_image(model_id='google/ddpm-bedroom-256', save_path='ddpm_generated_bedroom.png'):
    """
    Generate a realistic bedroom interior image using a pre-trained DDPM model.

    Args:
        model_id (str): The identifier for the pre-trained model.
        save_path (str): The file path where the generated image will be saved.

    Returns:
        None: The function saves the generated image to the specified path.
    """
    # Load the pre-trained Denoising Diffusion Probabilistic Model
    ddpm = DDPMPipeline.from_pretrained(model_id)

    # Generate the image
    image = ddpm().images[0]

    # Save the generated image
    image.save(save_path)


# test_function_code --------------------

def test_generate_bedroom_image():
    print("Testing started.")

    # Test case 1: Check if function generates an image file
    print("Testing case [1/1] started.")
    generate_bedroom_image()
    generated_image_path = 'ddpm_generated_bedroom.png'
    assert os.path.exists(generated_image_path), f"Test case [1/1] failed: {generated_image_path} does not exist."
    os.remove(generated_image_path)  # Clean up generated file after test

    print("Testing finished.")

    # Run the test function
    test_generate_bedroom_image()
