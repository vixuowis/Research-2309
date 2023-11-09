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
        image_features (torch.Tensor): The features of the image extracted by the ViT model.

    Raises:
        Exception: If the image cannot be opened.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        print(f'Error: {e}')
        return

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    image_features = outputs.last_hidden_state

    return image_features

# test_function_code --------------------

def test_generate_hashtags():
    """
    Test the 'generate_hashtags' function with a sample image URL.
    """
    image_url = 'https://example-image-url.com/image.jpg'
    image_features = generate_hashtags(image_url)
    assert image_features is not None, 'No features were extracted from the image.'
    assert image_features.size(0) == 1, 'The number of images processed should be 1.'
    assert image_features.size(1) == 768, 'The size of the image features should be 768.'

# call_test_function_code --------------------

test_generate_hashtags()