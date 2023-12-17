# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_image(image):
    """
    Classify an image as either a bike or a car using zero-shot classification.

    Parameters:
    image (PIL.Image): The image to classify.

    Returns:
    dict: Classification results with keys as class names and values as confidence scores.
    """
    class_names = ['bike', 'car']
    clip = pipeline('zero-shot-classification', model='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind')
    result = clip(image, class_names)
    return result

# test_function_code --------------------

def test_classify_image():
    print("Testing classify_image function.")
    # Assuming an image is loaded using PIL's Image open method
    image_bike = Image.open('test_bike.jpg')
    image_car = Image.open('test_car.jpg')

    # Test classification of a bike image
    result_bike = classify_image(image_bike)
    assert 'bike' in result_bike['labels'], f"Bike classification failed, result: {result_bike}"

    # Test classification of a car image
    result_car = classify_image(image_car)
    assert 'car' in result_car['labels'], f"Car classification failed, result: {result_car}"

    print("All tests passed!")

# Perform the test
test_classify_image()