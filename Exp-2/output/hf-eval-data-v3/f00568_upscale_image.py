# function_import --------------------

from transformers import Swin2SRForImageSuperResolution
from PIL import Image

# function_code --------------------

def upscale_image(image_path: str, model_path: str = 'caidas/swin2sr-classical-sr-x2-64') -> None:
    """
    Upscale an image using a pre-trained model from Hugging Face Transformers.

    Args:
        image_path (str): The path to the image to be upscaled.
        model_path (str, optional): The path to the pre-trained model. Defaults to 'caidas/swin2sr-classical-sr-x2-64'.

    Raises:
        UnidentifiedImageError: If the image file cannot be identified.
    """
    image = Image.open(image_path)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_path)
    upscaled_image = model(image)
    upscaled_image.save('upscaled_' + image_path)

# test_function_code --------------------

def test_upscale_image():
    """Test the upscale_image function."""
    try:
        upscale_image('test_image.jpg')
        print('Test passed')
    except Exception as e:
        print('Test failed. ' + str(e))

# call_test_function_code --------------------

test_upscale_image()