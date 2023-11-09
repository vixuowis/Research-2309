# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_device_image(image_path: str, class_names: list = ['smartphone', 'laptop', 'tablet']) -> dict:
    """
    Classify the device in the given image using the Hugging Face's pipeline function.

    Args:
        image_path (str): Path to the image file.
        class_names (list, optional): List of class names. Defaults to ['smartphone', 'laptop', 'tablet'].

    Returns:
        dict: The predicted class and the confidence score.
    """
    device_classifier = pipeline('image-classification', model='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft')
    device_class_prediction = device_classifier(image_path, class_names)
    return device_class_prediction

# test_function_code --------------------

def test_classify_device_image():
    """
    Test the classify_device_image function.
    """
    image_path = 'path/to/test_image.jpg'
    class_names = ['smartphone', 'laptop', 'tablet']
    prediction = classify_device_image(image_path, class_names)
    assert isinstance(prediction, dict), 'The result should be a dictionary.'
    assert 'label' in prediction, 'The result should have a label.'
    assert 'score' in prediction, 'The result should have a score.'

# call_test_function_code --------------------

test_classify_device_image()