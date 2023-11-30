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
    
    try:
        # Create feature extractor and model.
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/regnet-x-0-4")
        model = RegNetForImageClassification.from_pretrained("facebook/regnet-x-0-4")
        
    except OSError as error:
        raise error
    
    else:
        # Get image from url, convert to numpy array and normalize pixel values.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.57 Safari/537.36 OPR/18.0.1284.49'
        }
        
        image = Image.open(BytesIO(requests.get(image_url, headers=headers).content)).convert('RGB')
        image = feature_extractor(images=image, return_tensors="pt").pixel_values
        image /= image.max()
        
        # Predict label and convert to string.
        prediction = model(image)
        predicted_label = torch.argmax(prediction.logits).item()
        labels_map = feature_extractor.get_feature_info()["metadata"]["labels"]
        predicted_class = labels_map[predicted_label].lower().replace(" ", "_")
    
    return predicted_class

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