# requirements_file --------------------

import subprocess

requirements = ["urllib3", "Pillow", "torch", "timm"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import torch
import timm

# function_code --------------------

def classify_image(image_url: str) -> int:
    “”“
    Classifies an image fetched from a given URL using the pre-trained MobileNetV3 model.

    Args:
        image_url (str): A URL to the image that needs to be classified.

    Returns:
        int: The class index of the predicted class for the input image.

    Raises:
        ValueError: If the image cannot be opened.
        RuntimeError: If the model fails to process the image.
    “”“

    # Load the image from the given URL
    try:
        img = Image.open(urlopen(image_url))
    except Exception as e:
        raise ValueError(f'Error opening image: {e}')

    # Load the pre-trained MobileNetV3 model
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    model.eval()

    # Resolve model data configuration and create the input transforms
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Apply the transformations to the input image
    input_tensor = transforms(img).unsqueeze(0)

    # Classify the image
    with torch.no_grad():
        output = model(input_tensor)
        predictions = torch.nn.functional.softmax(output, dim=1)

    # Get the highest probability class
    predicted_class = predictions.argmax(dim=1).item()

    return predicted_class

# test_function_code --------------------

def test_classify_image():
    print("Testing started.")
    image_url = 'https://example.com/test_image.jpg'

    # Testing case 1: Check if the function returns an integer.
    print("Testing case [1/1] started.")
    predicted_class = classify_image(image_url)
    assert isinstance(predicted_class, int), "Test case [1/1] failed: The function should return an integer."
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_image()