# requirements_file --------------------

import subprocess

requirements = ["diffusers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_minecraft_skin():
    """
    Generates a unique Minecraft skin image using a diffusion model.

    Args:
        None

    Returns:
        An image object of the generated Minecraft skin in RGBA format.

    Raises:
        RuntimeError: If any error occurs while loading the model or generating the image.
    """
    try:
        pipeline = DDPMPipeline.from_pretrained('WiNE-iNEFF/Minecraft-Skin-Diffusion-V2')
        image = pipeline().images[0].convert('RGBA')
        return image
    except Exception as e:
        raise RuntimeError('Error generating Minecraft skin: ' + str(e))

# test_function_code --------------------

def test_generate_minecraft_skin():
    print("Testing started.")
    # No additional data set-up required

    # Test case 1: Generate a single skin
    print("Testing case [1/1] started.")
    image = generate_minecraft_skin()
    assert image.mode == 'RGBA', f"Test case [1/1] failed: Expected image mode 'RGBA', got {image.mode}."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_minecraft_skin()