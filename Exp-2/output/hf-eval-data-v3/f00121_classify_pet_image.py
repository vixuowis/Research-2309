# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_pet_image(image_path: str, labels: list = ['cat', 'dog']) -> dict:
    """
    Classify the pet image using a pretrained model.

    Args:
        image_path (str): The path to the image file.
        labels (list, optional): The list of possible class names. Defaults to ['cat', 'dog'].

    Returns:
        dict: The classification result.
    """
    clip = pipeline('image-classification', model='laion/CLIP-convnext_base_w-laion2B-s13B-b82K')
    classification_result = clip(image_path, labels)
    return classification_result

# test_function_code --------------------

def test_classify_pet_image():
    """
    Test the function classify_pet_image.
    """
    # Test case 1: Classify a cat image
    cat_image_path = 'https://placekitten.com/200/300'
    cat_result = classify_pet_image(cat_image_path)
    assert isinstance(cat_result, dict)
    assert 'label' in cat_result
    assert 'score' in cat_result

    # Test case 2: Classify a dog image
    dog_image_path = 'https://placedog.net/500'
    dog_result = classify_pet_image(dog_image_path)
    assert isinstance(dog_result, dict)
    assert 'label' in dog_result
    assert 'score' in dog_result

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_pet_image()