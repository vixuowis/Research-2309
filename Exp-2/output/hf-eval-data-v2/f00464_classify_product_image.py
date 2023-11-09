# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm

# function_code --------------------

def classify_product_image(url):
    """
    Classify a product image into relevant categories using a pretrained MobileNet-v3 model.

    Args:
        url (str): The URL of the product image.

    Returns:
        Tensor: A tensor containing the category probabilities output by the model.
    """
    img = Image.open(urlopen(url))
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    output = model(transforms(img).unsqueeze(0))

    return output

# test_function_code --------------------

def test_classify_product_image():
    """
    Test the classify_product_image function.
    """
    url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    output = classify_product_image(url)
    assert output is not None, 'The output should not be None.'
    assert output.size(0) == 1, 'The output should have a batch size of 1.'

# call_test_function_code --------------------

test_classify_product_image()