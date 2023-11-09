# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_image(image_path: str, class_names: str = 'cat, dog, bird') -> dict:
    """
    Classify an image into one of the given classes using a pre-trained model.

    Args:
        image_path (str): The path to the image to be classified.
        class_names (str): A string of class names separated by commas. Default is 'cat, dog, bird'.

    Returns:
        dict: A dictionary containing the predicted class and the confidence score.
    """
    model = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    results = model(image_path, class_names=class_names)
    return results

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function.
    """
    image_path = 'test_image.jpg'  # Replace with the path to a test image
    expected_class = 'dog'  # Replace with the expected class of the test image
    results = classify_image(image_path)
    assert expected_class in results, f'Expected class {expected_class}, but got {results}'

# call_test_function_code --------------------

test_classify_image()