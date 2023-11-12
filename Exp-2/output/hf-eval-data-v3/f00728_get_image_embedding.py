# function_import --------------------

from vc_models.models.vit import model_utils
import torch
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def get_image_embedding(image_url: str) -> torch.Tensor:
    """
    Get the image embedding from a pretrained model.

    Args:
        image_url (str): The url of the image to be processed.

    Returns:
        torch.Tensor: The image embedding.

    Raises:
        Exception: If the image cannot be loaded or the model cannot process the image.
    """
    try:
        # Load the pretrained model
        model, embd_size, model_transforms, model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)

        # Load the image from the url
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Transform the image
        transformed_img = model_transforms(img)

        # Get the image embedding
        embedding = model(transformed_img)

        return embedding
    except Exception as e:
        print(f'Error: {e}')
        raise

# test_function_code --------------------

def test_get_image_embedding():
    """
    Test the get_image_embedding function.
    """
    try:
        # Test with a valid image url
        embedding1 = get_image_embedding('https://placekitten.com/200/300')
        assert isinstance(embedding1, torch.Tensor), 'The output should be a torch.Tensor'

        # Test with another valid image url
        embedding2 = get_image_embedding('https://placekitten.com/400/600')
        assert isinstance(embedding2, torch.Tensor), 'The output should be a torch.Tensor'

        # Test with an invalid image url
        try:
            get_image_embedding('invalid_url')
        except Exception:
            pass
        else:
            assert False, 'An exception should be raised for an invalid url'

        print('All Tests Passed')
    except Exception as e:
        print(f'Test failed: {e}')
        raise

# call_test_function_code --------------------

test_get_image_embedding()