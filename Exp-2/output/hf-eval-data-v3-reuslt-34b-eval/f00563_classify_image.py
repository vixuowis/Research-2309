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
    processor = AutoImageProcessor.from_pretrained(pretrained_vision_model)
    
    with Image.open(requests.get(image_url, stream=True).raw) as image:
        inputs = processor(images=image, return_tensors="pt")
        
    model = AutoModelForImageClassification.from_pretrained(pretrained_vision_model)
    
    outputs = model(**inputs)
    
    probabilities = outputs.logits.softmax(-1)[0] # We use the first image in the batch.
    
    predicted_class_idx = probabilities.argmax().item()
    
    return imagenet_labels[predicted_class_idx]

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