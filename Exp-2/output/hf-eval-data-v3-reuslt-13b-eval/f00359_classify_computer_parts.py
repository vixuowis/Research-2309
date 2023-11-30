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

    processor = ViTImageProcessor(is_training=False, image_size=384)
    model = ViTForImageClassification.from_pretrained("hustee/computer-parts")
    
    with Image.open(user_uploaded_image_file_path) as img:
        processed_img = processor(images=img, return_tensors="pt")[0] # batch size is one by default
        
    predicted_label = model(**processed_img).logits.argmax().item()
    
    labels_dict = {0: "cpu", 1: "motherboard", 2: "graphics card", 3: "memory", 4: "hard disk drive", 5: "optical disk drive", 6: "power supply"}

    return labels_dict[predicted_label] # predicted label of the computer part in the image


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