# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_food_images(image_path, food_classes=['pizza', 'sushi', 'sandwich', 'salad', 'cake']):
    """
    Classify food images using a pre-trained model.

    Args:
        image_path (str): The path to the image to be classified.
        food_classes (list, optional): A list of possible food classes. Defaults to ['pizza', 'sushi', 'sandwich', 'salad', 'cake'].

    Returns:
        dict: The classification results.
    """
    image_classifier = pipeline('image-classification', model='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    result = image_classifier(image_path, possible_class_names=food_classes)
    return result

# test_function_code --------------------

def test_classify_food_images():
    """
    Test the classify_food_images function.
    """
    image_path = 'test_image.jpg'  # replace with your test image path
    food_classes = ['pizza', 'sushi', 'sandwich', 'salad', 'cake']
    result = classify_food_images(image_path, food_classes)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'label' in result, 'The result should contain a label.'
    assert 'score' in result, 'The result should contain a score.'

# call_test_function_code --------------------

test_classify_food_images()