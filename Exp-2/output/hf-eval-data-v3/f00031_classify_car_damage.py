# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_car_damage(image_path: str, class_names: list = ['major accident', 'minor damages']) -> dict:
    '''
    Classify the severity of car damage using a pre-trained model from Hugging Face.

    Args:
        image_path (str): Path to the image file or URL.
        class_names (list, optional): List of possible class names. Defaults to ['major accident', 'minor damages'].

    Returns:
        dict: The classification result indicating whether the car has been involved in a major accident or had minor damages.
    '''
    classifier = pipeline('image-classification', model='laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    result = classifier(image_path, class_names)
    return result

# test_function_code --------------------

def test_classify_car_damage():
    '''
    Test the function classify_car_damage.
    '''
    # Test case 1: Image of a car with major damage
    image_path1 = 'https://example.com/major_damage.jpg'
    result1 = classify_car_damage(image_path1)
    assert result1['label'] in ['major accident', 'minor damages']

    # Test case 2: Image of a car with minor damage
    image_path2 = 'https://example.com/minor_damage.jpg'
    result2 = classify_car_damage(image_path2)
    assert result2['label'] in ['major accident', 'minor damages']

    # Test case 3: Image of a car with no visible damage
    image_path3 = 'https://example.com/no_damage.jpg'
    result3 = classify_car_damage(image_path3)
    assert result3['label'] in ['major accident', 'minor damages']

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_car_damage()