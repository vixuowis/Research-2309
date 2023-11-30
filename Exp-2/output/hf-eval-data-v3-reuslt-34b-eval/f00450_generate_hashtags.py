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

    # Define pre-trained Vision Transformer model
    processor = ViTImageProcessor()
    model = ViTModel.from_pretrained('google/vit-base-patch16-224')

    # Get image from URL and convert to tensor
    try:
        img = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise Exception('Cannot open image from URL: ' + str(e))
    
    with processor.as_target_processor():
        inputs = processor([img], return_tensors='pt')

    # Extract features
    image_features = model(**inputs).last_hidden_state[0]

    # Normalize and sum across rows to get one vector by image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.sum(dim=0)
    image_features /= image_features.norm()

    # Return tensor of features
    return image_features

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