# requirements_file --------------------

!pip install -U urllib PIL torch timm

# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import torch
import timm

# function_code --------------------

def classify_image(image_url):
    """
    Classify an image from a given URL into a thousand categories.

    Args:
        image_url (str): The URL of the image to classify.

    Returns:
        torch.Tensor: The softmax probabilities for each of the 1,000 categories.

    Raises:
        ValueError: If the image URL is invalid or cannot be opened.
    """
    # Load the pretrained ConvNeXt model
    model = timm.create_model('convnext_base.fb_in1k', pretrained=True)
    model.eval()

    # Obtain the data configuration and transformations for the model
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config)

    # Load the image from the URL
    try:
        img = Image.open(urlopen(image_url))
    except Exception as e:
        raise ValueError(f'Cannot open image URL: {image_url}') from e

    # Transform the image and add a batch dimension
    img = transform(img).unsqueeze(0)

    # Classify the image
    with torch.no_grad():
        output = model(img)

    # Return the softmax probabilities
    return torch.nn.functional.softmax(output, dim=1)

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    image_urls = [
        'https://example.com/valid_image.jpg',  # Valid image URL
        'https://example.com/invalid_image.jpg',  # Invalid image URL
        ''  # Empty URL
    ]

    # Test case 1: Valid image URL
    print("Testing case [1/3] started.")
    try:
        probs = classify_image(image_urls[0])
        assert probs.shape == (1, 1000), f"Test case [1/3] failed: Expected output shape (1, 1000) but got {probs.shape}"
    except Exception as e:
        print(f"Exception occurred: {e}")

    # Test case 2: Invalid image URL
    print("Testing case [2/3] started.")
    try:
        classify_image(image_urls[1])
        print("Test case [2/3] should have failed but passed.")
    except ValueError:
        print("Test case [2/3] passed.")

    # Test case 3: Empty URL
    print("Testing case [3/3] started.")
    try:
        classify_image(image_urls[2])
        print("Test case [3/3] should have failed but passed.")
    except ValueError:
        print("Test case [3/3] passed.")

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image()