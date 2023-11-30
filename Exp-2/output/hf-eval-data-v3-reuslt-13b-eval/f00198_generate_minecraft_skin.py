# function_import --------------------

from diffusers import DDPMPipeline
from PIL import Image

# function_code --------------------

def generate_minecraft_skin():
    """
    This function generates a Minecraft skin image using a pre-trained model from Hugging Face Transformers.

    Returns:
        PIL.Image.Image: The generated Minecraft skin image in RGBA format.
    """
    # Load a pretrained pytorch lightning module that is compatible with diffusers
    ddpm_model = DDPMPipeline.load_from_checkpoint(path="https://github.com/huggingface/transformers-custom-apis/raw/ddpmpipelining/examples/tutorials/diffusion%20models/assets/ddpm-minecraft-128x128.ckpt")
    
    # Generate the image
    generated_image = ddpm_model.generate_images(inputs=None, num_imgs=1, batch_size=1)
    print("Generated Image Shape: ",generated_image[0].shape)

    # Convert it into a PIL Image
    pil_image = Image.fromarray(generated_image[0])
    return pil_image


# test_function_code --------------------

def test_generate_minecraft_skin():
    """
    This function tests the generate_minecraft_skin function by checking the type and mode of the returned image.
    """
    image = generate_minecraft_skin()
    assert isinstance(image, Image.Image), 'The returned object is not a PIL image.'
    assert image.mode == 'RGBA', 'The image is not in RGBA format.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_minecraft_skin()