# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm

# function_code --------------------

def classify_product_image(image_url):
    """
    Classify a product image into relevant categories using a pretrained MobileNet-v3 model.

    Args:
        image_url (str): The URL of the product image.

    Returns:
        torch.Tensor: The output tensor from the model, representing category probabilities.

    Raises:
        URLError: If the image_url is not accessible.
        IOError: If there is an error in opening the image.
    """
    img = Image.open(urlopen(image_url))
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    output = model(transforms(img).unsqueeze(0))

    return output

# test_function_code --------------------

def test_classify_product_image():
    """
    Test the classify_product_image function with some test cases.
    """
    # Test case 1: A product image of a cat
    output1 = classify_product_image('https://placekitten.com/200/300')
    assert output1 is not None, 'Test Case 1 Failed'

    # Test case 2: A product image of a dog
    output2 = classify_product_image('https://placedog.net/500')
    assert output2 is not None, 'Test Case 2 Failed'

    # Test case 3: A product image of a car
    output3 = classify_product_image('https://dummyimage.com/600x400/000/fff&text=car')
    assert output3 is not None, 'Test Case 3 Failed'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_product_image()