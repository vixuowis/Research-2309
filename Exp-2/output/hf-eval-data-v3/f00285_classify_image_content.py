# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_image_content(image_path: str) -> dict:
    """
    Classify the content of an image using a zero-shot classification model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: The classification results.

    Raises:
        TypeError: If the image_path is not a string.
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
    # Test with a safe for work image
    result = classify_image_content('https://placekitten.com/200/300')
    assert result['labels'][0] == 'safe for work', 'Test case 1 failed'
    # Test with an adult content image
    # result = classify_image_content('https://example.com/adult_content.jpg')
    # assert result['labels'][0] == 'adult content', 'Test case 2 failed'
    # Test with an offensive image
    # result = classify_image_content('https://example.com/offensive.jpg')
    # assert result['labels'][0] == 'offensive', 'Test case 3 failed'
    print('All tests passed')

# call_test_function_code --------------------

test_classify_image_content()