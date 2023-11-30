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

    # Load the model
    processor = AutoImageProcessor.from_pretrained('google/mobilenet_v1_0.75_192')
    model = AutoModelForImageClassification.from_pretrained('google/mobilenet_v1_0.75_192', num_labels=5,
                                                            label2id={'cute': 0, 'likeable': 1, 'good': 2,
                                                                      'beautiful': 3, 'none': 4})
    # Load the image from url
    response = requests.get(image_url)
    img = Image.open(Image.io.BytesIO(response.content))
    
    # Classify it
    preds = processor(images=img, return_tensors="pt")
    outputs = model(preds["pixel_values"])[0].tolist()
    predicted_class_idx = int(outputs.index(max(outputs)))
    
    # Return the predicted class
    if predicted_class_idx == 4: return "none"
    elif predicted_class_idx <= 2: return 'good'
    else: return {0:'cute',1:'likeable',2:'beautiful'}[predicted_class_idx%3]

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