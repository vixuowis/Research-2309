# function_import --------------------

from transformers import Swin2SRForConditionalGeneration
import torch
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def upscale_image(image_url: str) -> Image:
    """
    Upscale a low-resolution image to twice its size using a pretrained Swin2SR model.

    Args:
        image_url (str): The URL of the low-resolution image to be upscaled.

    Returns:
        Image: The upscaled image.

    Raises:
        Exception: If the image cannot be loaded from the provided URL.
    """
    # Load the pretrained model
    model = Swin2SRForConditionalGeneration.from_pretrained('conde/Swin2SR-lightweight-x2-64')

    # Load the image from the URL
    response = requests.get(image_url)
    low_resolution_image = Image.open(BytesIO(response.content))

    # Convert the image to a PyTorch tensor
    low_resolution_tensor = torch.tensor(np.array(low_resolution_image)).unsqueeze(0)

    # Pass the tensor through the model to obtain the upscaled image
    upscaled_tensor = model(low_resolution_tensor)

    # Convert the upscaled tensor back to an image
    upscaled_image = Image.fromarray(upscaled_tensor.squeeze(0).numpy())

    return upscaled_image

# test_function_code --------------------

def test_upscale_image():
    """
    Test the upscale_image function with a few test cases.
    """
    # Test case 1: A small image
    image_url = 'https://placekitten.com/200/300'
    upscaled_image = upscale_image(image_url)
    assert upscaled_image.size == (400, 600), 'Test case 1 failed'

    # Test case 2: A medium-sized image
    image_url = 'https://placekitten.com/500/500'
    upscaled_image = upscale_image(image_url)
    assert upscaled_image.size == (1000, 1000), 'Test case 2 failed'

    # Test case 3: A large image
    image_url = 'https://placekitten.com/1000/1000'
    upscaled_image = upscale_image(image_url)
    assert upscaled_image.size == (2000, 2000), 'Test case 3 failed'

    return 'All tests passed'

# call_test_function_code --------------------

test_upscale_image()