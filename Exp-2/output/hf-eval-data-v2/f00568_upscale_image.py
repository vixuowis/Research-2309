# function_import --------------------

from transformers import Swin2SRForImageSuperResolution
from PIL import Image

# function_code --------------------

def upscale_image(image_path: str, model_name: str = 'caidas/swin2sr-classical-sr-x2-64') -> None:
    """
    Upscale an image using a pre-trained model from Hugging Face Transformers.

    Args:
        image_path (str): The path to the image to be upscaled.
        model_name (str, optional): The name of the pre-trained model to use for upscaling. Defaults to 'caidas/swin2sr-classical-sr-x2-64'.

    Returns:
        None. The upscaled image is saved in the same directory as the original image, with '_upscaled' appended to the original filename.
    """
    # Load the image
    image = Image.open(image_path)

    # Load the pre-trained model
    model = Swin2SRForImageSuperResolution.from_pretrained(model_name)

    # Upscale the image
    upscaled_image = model(image)

    # Save the upscaled image
    upscaled_image_path = image_path.rsplit('.', 1)[0] + '_upscaled.' + image_path.rsplit('.', 1)[1]
    upscaled_image.save(upscaled_image_path)

# test_function_code --------------------

def test_upscale_image():
    """
    Test the upscale_image function.

    Raises:
        AssertionError: If the function does not correctly upscale the image.
    """
    # Define the path to a test image
    test_image_path = 'test_image.jpg'

    # Upscale the test image
    upscale_image(test_image_path)

    # Check that the upscaled image exists
    upscaled_image_path = test_image_path.rsplit('.', 1)[0] + '_upscaled.' + test_image_path.rsplit('.', 1)[1]
    assert os.path.exists(upscaled_image_path), 'Upscaled image does not exist.'

# call_test_function_code --------------------

test_upscale_image()