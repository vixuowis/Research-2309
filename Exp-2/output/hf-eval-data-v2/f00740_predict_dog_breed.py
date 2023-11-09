# function_import --------------------

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch

# function_code --------------------

def predict_dog_breed(user_uploaded_image):
    """
    This function takes an image as input and predicts the breed of the dog in the image.
    
    Args:
        user_uploaded_image (PIL.Image): The image of the dog uploaded by the user.
    
    Returns:
        str: The predicted breed of the dog in the image.
    """
    feature_extractor = ConvNextFeatureExtractor.from_pretrained('facebook/convnext-tiny-224')
    model = ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224')
    inputs = feature_extractor(user_uploaded_image, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    dog_breed = model.config.id2label[predicted_label]
    return dog_breed

# test_function_code --------------------

def test_predict_dog_breed():
    """
    This function tests the predict_dog_breed function by using a sample image of a dog.
    """
    from PIL import Image
    import requests
    from io import BytesIO
    
    url = 'https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    predicted_breed = predict_dog_breed(img)
    
    assert isinstance(predicted_breed, str), 'The prediction should be a string.'

# call_test_function_code --------------------

test_predict_dog_breed()