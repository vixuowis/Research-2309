# function_import --------------------

from transformers import pipeline

# function_code --------------------

def identify_smartphone_brand_and_luxury_level(image_path: str, class_names: str = 'Apple, Samsung, Huawei, Xiaomi, low luxury level, medium luxury level, high luxury level') -> dict:
    """
    Identify which smartphone brand is featured in an image and predict the intensity of luxury level.

    Args:
        image_path (str): Path to the image.
        class_names (str): Comma-separated list of possible class names. Default is 'Apple, Samsung, Huawei, Xiaomi, low luxury level, medium luxury level, high luxury level'.

    Returns:
        dict: The predicted class label, indicating the smartphone brand and the intensity of luxury level of the given image.
    """
    image_classification = pipeline('image-classification', model='laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg')
    result = image_classification(image_path, class_names)
    return result

# test_function_code --------------------

def test_identify_smartphone_brand_and_luxury_level():
    """
    Test function identify_smartphone_brand_and_luxury_level.
    """
    image_path = 'https://placekitten.com/200/300'
    result = identify_smartphone_brand_and_luxury_level(image_path)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'label' in result, 'The result should contain a label.'
    assert 'score' in result, 'The result should contain a score.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_identify_smartphone_brand_and_luxury_level()