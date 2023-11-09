# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_image_content(image_path):
    """
    Classify the content of an image using a zero-shot classification model.

    Args:
        image_path (str): The path to the image file or an image URL.

    Returns:
        dict: The classification results. Each key is a class name and the corresponding value is the confidence score.
    """
    image_classifier = pipeline('zero-shot-classification', model='laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    class_names = ['safe for work', 'adult content', 'offensive']
    result = image_classifier(image=image_path, class_names=class_names)
    return result

# test_function_code --------------------

def test_classify_image_content():
    """
    Test the classify_image_content function.
    """
    image_path = 'https://example.com/test_image.jpg'
    result = classify_image_content(image_path)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert set(result.keys()) == set(['safe for work', 'adult content', 'offensive']), 'The result should contain the predefined class names.'
    for score in result.values():
        assert 0 <= score <= 1, 'Each score should be a float between 0 and 1.'

# call_test_function_code --------------------

test_classify_image_content()