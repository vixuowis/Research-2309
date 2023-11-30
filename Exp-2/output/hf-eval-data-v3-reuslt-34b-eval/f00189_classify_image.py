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
    
    # download the image from the given URL and load it into a PIL Image object
    with urlopen(img_url) as stream:
        img = Image.open(stream)
        
    # preprocess the image for the model
    model = timm.create_model("efficientnet_b0", pretrained=True).eval()
    inputs = torch.cat([model.preprocess_image(img)]).unsqueeze(0)
    
    # classify the image and return the model outputs
    with torch.no_grad():
        outputs = model(inputs)
    return outputs

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