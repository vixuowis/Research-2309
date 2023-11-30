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
    try:
        # load processor and model --------------------
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        processor = ViTImageProcessor(feature_extractor="google/vit-base-patch16-224", device=device)
        model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')

        # process image --------------------
        
        # download the image from the url
        response = requests.get(image_url, allow_redirects=True)
        if response.status_code != 200:
            raise OSError("Image file could not be loaded.")
        open('input_img/test.jpg', 'wb').write(response.content) # save the image locally
        
        # load the downloaded image to a PIL Image object
        img = Image.open('input_img/test.jpg') 
        
        # process the images and prepare for classification
        img_input, _ = processor(images=img)
        
        # classify the image --------------------
        
        # move to device before classifying
        model.to(device)
        img_input = img_input.to(device)

        # make a prediction
        pred = model(img_input, labels=None) 
        
        # get the top predicted class (0 is the first item in the list)
        preds = torch.nn.functional.softmax(pred[0], dim=0)
        predicted_class_idx = preds.argsort()[-1].item()
        predicted_class = model.config.id2label[predicted_class_idx]
        
    except OSError:
        raise OSError("Image file could not be loaded.")
    
    return predicted_class


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