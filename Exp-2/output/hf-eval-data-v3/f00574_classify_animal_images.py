# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_animal_images(image_path: str, categories: list) -> dict:
    """
    Classify images of animals into their specific categories using a pre-trained model.

    Args:
        image_path (str): Path to the image file.
        categories (list): List of categories to classify the image into.

    Returns:
        dict: A dictionary containing the predicted category and its confidence score.

    Raises:
        OSError: If the model file is not found.
    """
    classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')
    result = classifier(image_path, categories)
    return result

# test_function_code --------------------

def test_classify_animal_images():
    """Test the classify_animal_images function."""
    image_path = 'path/to/test/image.jpg'
    categories = ['cat', 'dog', 'bird', 'fish']
    result = classify_animal_images(image_path, categories)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'label' in result, 'The result should contain a label.'
    assert 'score' in result, 'The result should contain a score.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_animal_images()