# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_medical_image(image_path: str, possible_class_names: list) -> dict:
    """
    Classify a medical image to find out if it's an X-ray, an MRI scan, or a CT scan.

    Args:
        image_path (str): The path to the medical image that you'd like to classify.
        possible_class_names (list): A list of possible class names corresponding to the types of scans (e.g., X-ray, MRI scan, CT scan).

    Returns:
        dict: The probabilities for each class. You can then select the class with the highest probability as the predicted class.
    """
    clip = pipeline('zero-shot-image-classification', model='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    result = clip(image_path, possible_class_names)
    return result

# test_function_code --------------------

def test_classify_medical_image():
    """
    Test the function classify_medical_image.
    """
    image_path = 'path/to/test_image.png'
    possible_class_names = ['X-ray', 'MRI scan', 'CT scan']
    result = classify_medical_image(image_path, possible_class_names)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'X-ray' in result, 'The result should contain the class name X-ray.'
    assert 'MRI scan' in result, 'The result should contain the class name MRI scan.'
    assert 'CT scan' in result, 'The result should contain the class name CT scan.'

# call_test_function_code --------------------

test_classify_medical_image()