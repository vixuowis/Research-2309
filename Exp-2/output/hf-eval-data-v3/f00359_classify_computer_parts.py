# function_import --------------------

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# function_code --------------------

def classify_computer_parts(user_uploaded_image_file_path):
    '''
    Classify the computer parts in the image uploaded by the user.

    Args:
        user_uploaded_image_file_path (str): The file path of the image uploaded by the user.

    Returns:
        str: The predicted label of the computer part in the image.
    '''
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    image = Image.open(user_uploaded_image_file_path)
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    return predicted_label

# test_function_code --------------------

def test_classify_computer_parts():
    '''
    Test the function classify_computer_parts.
    '''
    url = 'https://placekitten.com/200/300'
    response = requests.get(url, stream=True)
    with open('test_image.jpg', 'wb') as f:
        f.write(response.content)

    predicted_label = classify_computer_parts('test_image.jpg')
    assert isinstance(predicted_label, str), 'The predicted label should be a string.'

    print('All Tests Passed')

# call_test_function_code --------------------

test_classify_computer_parts()