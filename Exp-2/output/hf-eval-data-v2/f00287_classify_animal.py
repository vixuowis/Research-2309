# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_animal(image_path):
    """
    Classify an animal in an image as a cat or a dog using a pre-trained model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary containing the predicted class and the confidence score.
    """
    # Create an image classification model
    image_classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')
    # Apply the classifier to an image file
    result = image_classifier(image_path, ['cat', 'dog'])
    return result

# test_function_code --------------------

def test_classify_animal():
    """
    Test the classify_animal function.
    """
    # Define a test image path
    test_image_path = 'path/to/test_image.jpg'
    # Call the classify_animal function
    result = classify_animal(test_image_path)
    # Assert that the result is a dictionary
    assert isinstance(result, dict)
    # Assert that the dictionary contains the keys 'class' and 'score'
    assert 'class' in result and 'score' in result

# call_test_function_code --------------------

test_classify_animal()