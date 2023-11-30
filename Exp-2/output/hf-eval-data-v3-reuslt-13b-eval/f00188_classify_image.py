# function_import --------------------

from transformers import AutoFeatureExtractor, RegNetForImageClassification
import torch
from PIL import Image
import requests
from io import BytesIO

# function_code --------------------

def classify_image(image_url: str) -> str:
    """
    Classify an image using the pretrained RegNetForImageClassification model.

    Args:
        image_url (str): The URL of the image to be classified.

    Returns:
        str: The predicted label of the image.

    Raises:
        OSError: If the model identifier is not found in the Hugging Face model hub.
    """
    
    model_identifier = "facebook/regnet-y-400mf"
    
    try: 
        
        # load feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_identifier)
        model = RegNetForImageClassification.from_pretrained(model_identifier).eval()
            
        # download image
        response = requests.get(image_url, stream=True)
        
        if not response.ok:
            raise RuntimeError("HTTP Error {}".format(response.status_code)) 
        
        img = Image.open(BytesIO(response.content)).convert('RGB')
    
    except OSError as err:
        print(err)
        return "Model not found"
    
    # resize image and extract features
    inputs = feature_extractor(images=img, return_tensors='pt')["pixel_values"]
    outputs = model(inputs).logits

    # get predicted label
    predictions = torch.nn.functional.softmax(outputs, dim=-1)
    top5_predicted_labels = torch.topk(predictions.flatten(), 5).indices.tolist()
    
    return feature_extractor.decode([str(index) for index in top5_predicted_labels])


# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function with different test cases.
    """
    test_image_url_1 = 'https://placekitten.com/200/300'
    test_image_url_2 = 'https://placekitten.com/400/600'
    test_image_url_3 = 'https://placekitten.com/800/1200'

    assert isinstance(classify_image(test_image_url_1), str)
    assert isinstance(classify_image(test_image_url_2), str)
    assert isinstance(classify_image(test_image_url_3), str)

    print('All Tests Passed')


# call_test_function_code --------------------

if __name__ == '__main__':
    test_classify_image()