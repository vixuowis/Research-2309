# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_vintage_images(num_images=1):
    """
    Generates a specified number of vintage images using a pretrained diffusion model.

    Args:
        num_images (int): The number of vintage images to generate. Default is 1.

    Returns:
        List[PIL.Image]: A list of PIL Image objects representing the generated vintage images.

    Raises:
        ValueError: If the number of images requested is less than 1.
    """
    if num_images < 1:
        raise ValueError('The number of images requested must be at least 1.')
    pipeline = DDPMPipeline.from_pretrained('pravsels/ddpm-ffhq-vintage-finetuned-vintage-3epochs')
    images = [pipeline().images[0] for _ in range(num_images)]
    return images

# test_function_code --------------------

def test_generate_vintage_images():
    print("Testing started.")

    # Test case 1: Generating one vintage image
    print("Testing case [1/3] started.")
    images = generate_vintage_images(1)
    assert len(images) == 1, f"Test case [1/3] failed: Expected 1 image, got {len(images)}"

    # Test case 2: Generating three vintage images
    print("Testing case [2/3] started.")
    images = generate_vintage_images(3)
    assert len(images) == 3, f"Test case [2/3] failed: Expected 3 images, got {len(images)}"

    # Test case 3: Invalid argument (0 images requested)
    print("Testing case [3/3] started.")
    try:
        images = generate_vintage_images(0)
    except ValueError as e:
        assert str(e) == 'The number of images requested must be at least 1.', f"Test case [3/3] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_vintage_images()