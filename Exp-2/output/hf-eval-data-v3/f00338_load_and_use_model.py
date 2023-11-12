# function_import --------------------

from vc_models.models.vit import model_utils
import torch

# function_code --------------------

def load_and_use_model(img):
    """
    Load the pre-trained VC-1 model and use it to obtain an embedding from an image.

    Args:
        img (PIL.Image): The image to be processed.

    Returns:
        torch.Tensor: The embedding obtained from the image.
    """
    model, embd_size, model_transforms, model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
    transformed_img = model_transforms(img)
    embedding = model(transformed_img)
    return embedding

# test_function_code --------------------

def test_load_and_use_model():
    """
    Test the function load_and_use_model.
    """
    from PIL import Image
    import requests
    from io import BytesIO

    # Test case 1: A random image from the internet
    url = 'https://placekitten.com/200/300'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    embedding1 = load_and_use_model(img)
    assert embedding1.shape[0] == embd_size

    # Test case 2: Another random image from the internet
    url = 'https://placekitten.com/400/600'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    embedding2 = load_and_use_model(img)
    assert embedding2.shape[0] == embd_size

    # Test case 3: Yet another random image from the internet
    url = 'https://placekitten.com/800/1200'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    embedding3 = load_and_use_model(img)
    assert embedding3.shape[0] == embd_size

    return 'All Tests Passed'

# call_test_function_code --------------------

test_load_and_use_model()