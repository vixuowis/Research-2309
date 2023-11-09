# function_import --------------------

from transformers import pipeline

# function_code --------------------

def identify_smartphone_brand_and_luxury_level(image_path):
    """
    Identify which smartphone brand is featured in an image and predict the intensity of luxury level.

    Args:
        image_path (str): Path to the image.

    Returns:
        dict: The predicted class label, indicating the smartphone brand and the intensity of luxury level of the given image.
    """
    image_classification = pipeline('image-classification', model='laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg')
    class_names = 'Apple, Samsung, Huawei, Xiaomi, low luxury level, medium luxury level, high luxury level'
    result = image_classification(image_path, class_names)
    return result

# test_function_code --------------------

def test_identify_smartphone_brand_and_luxury_level():
    """
    Test the function identify_smartphone_brand_and_luxury_level.
    """
    image_path = 'path/to/test/image.jpg'
    result = identify_smartphone_brand_and_luxury_level(image_path)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'label' in result, 'The result should contain a label.'
    assert 'score' in result, 'The result should contain a score.'

# call_test_function_code --------------------

test_identify_smartphone_brand_and_luxury_level()