# requirements_file --------------------

!pip install -U diffusers pil

# function_import --------------------

from diffusers import DDPMPipeline


# function_code --------------------

def generate_celebrity_face(model_id='google/ddpm-ema-celebahq-256'):
    # Create a Denoising Diffusion Probabilistic Models (DDPM) pipeline for image synthesis
    ddpm = DDPMPipeline.from_pretrained(model_id)

    # Generate a random image
    generated_image = ddpm().images[0]

    # Save the generated image
    generated_image.save('generated_celebrity_face.png')
    return generated_image

# test_function_code --------------------

def test_generate_celebrity_face():
    print("Testing generate_celebrity_face function.")

    # Test case: Generate an image and check if an image object is returned
    print("Test case: Generate an image")
    generated_image = generate_celebrity_face()
    assert isinstance(generated_image, Image.Image), f"Failed to generate an image object."

    print("Testing finished.")

# Run the test function
test_generate_celebrity_face()
