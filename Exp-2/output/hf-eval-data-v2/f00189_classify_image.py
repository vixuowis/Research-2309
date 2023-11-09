# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm

# function_code --------------------

def classify_image(img_url):
    """
    Classify an image from a URL into a thousand categories using a pretrained ConvNeXt model.

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
    return model(transforms(img).unsqueeze(0))

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function with a sample image URL.
    """
    img_url = 'https://example.com/image.jpg'
    output = classify_image(img_url)
    assert output.size() == (1, 1000), 'The output size should be (1, 1000) for a single image.'

# call_test_function_code --------------------

test_classify_image()