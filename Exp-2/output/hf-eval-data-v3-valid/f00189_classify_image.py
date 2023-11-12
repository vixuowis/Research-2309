# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import torch
import timm

# function_code --------------------

def classify_image(img_url):
    """
    Classify an image from a URL into a thousand categories.

    Args:
        img_url (str): The URL of the image to be classified.

    Returns:
        torch.Tensor: The output of the model containing the probabilities for each of the 1,000 categories.
    """
    img = Image.open(urlopen(img_url))
    model = timm.create_model('convnext_base.fb_in1k', pretrained=True)
    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    output = model(transforms(img).unsqueeze(0))
    return output

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.
    """
    img_url = 'https://placekitten.com/200/300'
    output = classify_image(img_url)
    assert isinstance(output, torch.Tensor), 'Output should be a torch.Tensor'
    assert output.size(0) == 1, 'Output tensor should have size 1 in the first dimension'
    assert output.size(1) == 1000, 'Output tensor should have size 1000 in the second dimension'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_image()