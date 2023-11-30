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

    # load model from Hugging Face model hub
    try:
        model_id = "datasets/justincao/regnet-y-16gf"  # just a dummy value, we don't need the real model yet
        model = RegNetForImageClassification.from_pretrained(model_id)
    except OSError:
        raise OSError('Model identifier not found in Hugging Face model hub')
    
    # load feature extractor from Hugging Face model hub
    feature_extractor = AutoFeatureExtractor.from_pretrained("google/regnet-y-16gf")

    # preprocess image and add batch dimension using the preprocessing function
    image = Image.open(BytesIO(requests.get(image_url).content))  # download image from URL
    inputs = feature_extractor(images=image, return_tensors="pt", normalize=True)
    
    # predict class label with model and get predicted probabilities using the argmax function
    labels = [l.strip() for l in open('./labels.txt', 'r').readlines()]  # load class labels from file
    outputs = model(**inputs)
    logits = outputs.logits[0].cpu().detach().numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=0).tolist()
    
    # get predicted class using argmax function (index of highest entry in probability vector)
    predicted_class = labels[int(torch.argmax(probabilities))]
    
    return f'Predicted class: {predicted_class} \n with probability: {round(max(probabilities), 4)*100}%'

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