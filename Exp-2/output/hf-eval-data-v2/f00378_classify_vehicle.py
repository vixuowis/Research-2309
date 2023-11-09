# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_vehicle(image):
    """
    Classify the given image as either a bike or a car using a pretrained model.

    Args:
        image (PIL.Image): The image to classify.

    Returns:
        dict: The classification results.
    """
    clip = pipeline('zero-shot-classification', model='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind')
    class_names = ['bike', 'car']
    result = clip(image, class_names)
    return result

# test_function_code --------------------

def test_classify_vehicle():
    """
    Test the classify_vehicle function.
    """
    # Load a test image (replace with actual path)
    image = Image.open('test_image.jpg')
    result = classify_vehicle(image)
    # Check that the result is a dictionary and contains the expected keys
    assert isinstance(result, dict)
    assert 'labels' in result
    assert 'scores' in result
    # Check that the labels are correct
    assert set(result['labels']) == set(['bike', 'car'])

# call_test_function_code --------------------

test_classify_vehicle()