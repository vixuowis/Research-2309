# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm
import torch

# function_code --------------------

def classify_image(img_url: str) -> int:
    """
    Classify an image using a pretrained MobileNet-v3 model.

    Args:
        img_url (str): The URL of the image to classify.

    Returns:
        int: The predicted class of the image.

    Raises:
        URLError: If the image cannot be opened from the provided URL.
        RuntimeError: If there is a problem running the model.
    """
    img = Image.open(urlopen(img_url))
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    input_tensor = transforms(img).unsqueeze(0)
    output = model(input_tensor)

    return torch.argmax(output).item()

# test_function_code --------------------

def test_classify_image():
    """Test the classify_image function."""
    assert isinstance(classify_image('https://placekitten.com/200/300'), int)
    assert isinstance(classify_image('https://placekitten.com/200/301'), int)
    assert isinstance(classify_image('https://placekitten.com/200/302'), int)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()