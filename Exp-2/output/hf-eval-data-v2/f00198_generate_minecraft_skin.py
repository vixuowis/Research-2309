# function_import --------------------

from diffusers import DDPMPipeline

# function_code --------------------

def generate_minecraft_skin():
    """
    This function generates a Minecraft skin image using a pre-trained model.

    Returns:
        PIL.Image.Image: An image object in RGBA format.
    """
    pipeline = DDPMPipeline.from_pretrained('WiNE-iNEFF/Minecraft-Skin-Diffusion-V2')
    image = pipeline().images[0].convert('RGBA')
    return image

# test_function_code --------------------

def test_generate_minecraft_skin():
    """
    This function tests the generate_minecraft_skin function by generating an image and checking its mode and size.
    """
    image = generate_minecraft_skin()
    assert image.mode == 'RGBA', 'Image mode must be RGBA.'
    assert image.size == (64, 64), 'Image size must be 64x64.'

# call_test_function_code --------------------

test_generate_minecraft_skin()