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
    
    # Get the model and processor (pretrained)
    try:
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    except OSError:
        print("There was a problem while fetching the model and processor!")
        
    
    # Open the image
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    except OSError:
        print("There was a problem with loading the image!")


    # Process the image and make the prediction
    try:
        processed_img = processor(img, return_tensors="pt")
        probabilities = model(**processed_img).logits[0].softmax(-1)
    except OSError:
        print("There was a problem while processing or predicting!")
        
    
    # Return the class of the image with top probability.
    return "The image belongs to {}.".format(model.config.id2label[probabilities.argmax().item()])

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