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
    # Load the ViT model and pretrained weights.
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    processor = ViTImageProcessor((224, 224))
    
    # Download image from URL, convert it to PIL Image and process it with the ViT Image Processor.
    try:
        r = requests.get(image_url)
        img = Image.open(r.raw).convert('RGB')
    except Exception as e:
        raise Exception("Error while downloading image from URL")
    
    # Extract features for ViT model.
    image_features = processor(images=img, return_tensors="pt").pixel_values
    
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