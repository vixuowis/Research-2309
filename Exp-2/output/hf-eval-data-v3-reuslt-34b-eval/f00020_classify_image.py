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

    # Load the model and labels.

    model = timm.create_model("mobilenetv3_small", pretrained=True)
    labels = [s[12:] for s in (line.strip() for line in
                               urlopen('https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/imagenet_classes.txt'))]

    # Load the image from the URL, classify it, and get the label of the predicted class.
    
    img = Image.open(urlopen(img_url))
    with torch.no_grad():
        inp = torch.unsqueeze(model(torch.Tensor(255 * img).permute([2, 0, 1]) / 255 - [0.485, 0.456, 0.406], dim=0), 0)
        outp = torch.softmax(model(inp), -1)[0]
    label_idx = int(torch.argmax(outp))
    
    return label_idx

# test_function_code --------------------

def test_classify_image():
    """Test the classify_image function."""
    assert isinstance(classify_image('https://placekitten.com/200/300'), int)
    assert isinstance(classify_image('https://placekitten.com/200/301'), int)
    assert isinstance(classify_image('https://placekitten.com/200/302'), int)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_image()