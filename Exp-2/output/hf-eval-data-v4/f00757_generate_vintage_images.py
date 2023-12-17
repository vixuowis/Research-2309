# requirements_file --------------------

!pip install -U diffusers

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_vintage_images(number_of_images=1):
    # Load the pretrained vintage image diffusion model
    pipeline = DDPMPipeline.from_pretrained('pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs')
    # Generate the specified number of vintage images
    generated_images = [pipeline().images[0] for _ in range(number_of_images)]
    return generated_images


# test_function_code --------------------

def test_generate_vintage_images():
    print("Testing generate_vintage_images function.")
    # Generate a single vintage image
    images = generate_vintage_images(1)
    assert len(images) == 1, f"Expected 1 image, got {len(images)}"
    print("Test case passed: 1 generated image")
    
    # Generate multiple vintage images
    more_images = generate_vintage_images(5)
    assert len(more_images) == 5, f"Expected 5 images, got {len(more_images)}"
    print("Test case passed: 5 generated images")
    
    print("All tests passed!")

# Run the test function
test_generate_vintage_images()
