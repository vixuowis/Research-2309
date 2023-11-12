# function_import --------------------

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_image(image_url):
    '''
    Classify the image using Vision Transformer (ViT).

    Args:
        image_url (str): The url of the image to be classified.

    Returns:
        str: The predicted class of the image.

    Raises:
        OSError: If there is a problem with the network connection or the image file.
    '''
    image = Image.open(requests.get(image_url, stream=True).raw)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# test_function_code --------------------

def test_classify_image():
    '''
    Test the classify_image function.
    '''
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    predicted_class = classify_image(test_image_url)
    assert isinstance(predicted_class, str), 'The predicted class should be a string.'
    print('All Tests Passed')

# call_test_function_code --------------------

if __name__ == '__main__':
    test_classify_image()