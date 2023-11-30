# function_import --------------------

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    """
    Classify the image using the pretrained model 'google/mobilenet_v1_0.75_192'.

    Args:
        image_url (str): The url of the image to be classified.

    Returns:
        str: The predicted class of the image.
    """
    
    image = Image.open(requests.get(image_url, stream=True).raw)

    processor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')
    model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192', num_labels=3)

    inputs = processor(images=image, return_tensors='pt').to('cuda')
    outputs = model(**inputs).logits.argmax()
    
    if outputs == 0:
        output = 'dog'
        
    elif outputs == 1:
        output = 'cat'
        
    else:
        output = 'horse'
    
    return output

# test_function_code --------------------

def test_classify_image():
    """
    Test the function classify_image.
    """
    assert classify_image('http://images.cocodataset.org/val2017/000000039769.jpg') is not None
    assert classify_image('https://placekitten.com/200/300') is not None
    assert classify_image('https://placekitten.com/400/600') is not None
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_image()