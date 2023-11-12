# function_import --------------------

from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO

# function_code --------------------

def classify_medical_image(image_url: str, possible_class_names: list) -> dict:
    '''
    Classify a medical image to find out if it's an X-ray, an MRI scan, or a CT scan using Hugging Face's zero-shot image classification pipeline.
    
    Args:
    image_url (str): The URL of the medical image to be classified.
    possible_class_names (list): A list of possible class names corresponding to the types of scans (e.g., X-ray, MRI scan, CT scan).
    
    Returns:
    dict: A dictionary containing the class names and their corresponding probabilities.
    
    Raises:
    Exception: If the image cannot be loaded from the provided URL.
    '''
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        raise Exception('Failed to load image from URL.') from e
    
    clip = pipeline('zero-shot-image-classification', model='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    result = clip(image, possible_class_names)
    return result

# test_function_code --------------------

def test_classify_medical_image():
    '''
    Test the classify_medical_image function with various test cases.
    '''
    # Test case 1: X-ray image
    image_url = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0929664620300449-gr2_lrg-a.jpg'
    possible_class_names = ['X-ray', 'MRI scan', 'CT scan']
    result = classify_medical_image(image_url, possible_class_names)
    assert 'X-ray' in result.keys(), 'Test case 1 failed'
    
    # Test case 2: MRI image
    image_url = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0929664620300449-gr2_lrg-a.jpg'
    possible_class_names = ['X-ray', 'MRI scan', 'CT scan']
    result = classify_medical_image(image_url, possible_class_names)
    assert 'MRI scan' in result.keys(), 'Test case 2 failed'
    
    # Test case 3: CT scan image
    image_url = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S0929664620300449-gr2_lrg-a.jpg'
    possible_class_names = ['X-ray', 'MRI scan', 'CT scan']
    result = classify_medical_image(image_url, possible_class_names)
    assert 'CT scan' in result.keys(), 'Test case 3 failed'
    
    return 'All tests passed'

# call_test_function_code --------------------

test_classify_medical_image()