# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_plant(image_path: str, labels: list) -> str:
    """
    Classify the type of plant in the image provided.

    Args:
        image_path (str): The path to the image file.
        labels (list): A list of possible class names.

    Returns:
        str: The name of the probable plant.

    Raises:
        OSError: If the model 'laion/CLIP-convnext_base_w-laion2B-s13B-b82K' does not exist or the image file does not exist.
    """
    try:
        clip = pipeline('image-classification', model='laion/CLIP-convnext_base_w-laion2B-s13B-b82K')
        plant_classifications = clip(image_path, labels)
        top_plant = plant_classifications[0]['label']
        return top_plant
    except Exception as e:
        raise OSError('Model or image file not found.') from e

# test_function_code --------------------

def test_classify_plant():
    """
    Test the function classify_plant.
    """
    # Test case 1: Test with an image of a rose
    image_path = 'https://example.com/rose.jpg'
    labels = ['rose', 'tulip', 'sunflower']
    result = classify_plant(image_path, labels)
    assert isinstance(result, str), 'The result should be a string.'

    # Test case 2: Test with an image of a tulip
    image_path = 'https://example.com/tulip.jpg'
    labels = ['rose', 'tulip', 'sunflower']
    result = classify_plant(image_path, labels)
    assert isinstance(result, str), 'The result should be a string.'

    # Test case 3: Test with an image of a sunflower
    image_path = 'https://example.com/sunflower.jpg'
    labels = ['rose', 'tulip', 'sunflower']
    result = classify_plant(image_path, labels)
    assert isinstance(result, str), 'The result should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_plant()