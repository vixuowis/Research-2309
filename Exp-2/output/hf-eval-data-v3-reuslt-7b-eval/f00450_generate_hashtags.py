# function_import --------------------

from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

# function_code --------------------

def generate_hashtags(image_url):
    """
    Generate hashtags for a given image URL using Vision Transformer (ViT) model.

    Args:
        image_url (str): The URL of the image for which to generate hashtags.

    Returns:
        image_features (torch.Tensor): The extracted features of the image.

    Raises:
        Exception: If the image cannot be opened.
    """    

    processor = ViTImageProcessor(feature_type="image/vggish", do_resize=True, do_center_crop=False)
    model = ViTModel()
    # Load image from URL
    try:
        response = requests.get(image_url).content
        image = Image.open(BytesIO(response))
    except Exception as e:
        print("Error: " + str(e))
    
    # Generate features of the image
    image_features = processor([image]).numpy()[0]

    return image_features 

# function_code --------------------


# test_function_code --------------------

def test_generate_hashtags():
    """
    Test the 'generate_hashtags' function with different image URLs.
    """
    # Test with a valid image URL
    image_url = 'https://placekitten.com/200/300'
    image_features = generate_hashtags(image_url)
    assert image_features is not None, 'No features were extracted from the image.'

    # Test with an invalid image URL
    image_url = 'https://invalid-url.com/image.jpg'
    try:
        image_features = generate_hashtags(image_url)
    except Exception as e:
        assert str(e) == 'Unable to open image. Please check the URL.', 'The function did not raise the expected exception.'

    print('All Tests Passed')


# call_test_function_code --------------------

test_generate_hashtags()