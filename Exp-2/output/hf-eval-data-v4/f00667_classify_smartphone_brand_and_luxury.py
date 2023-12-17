# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_smartphone_brand_and_luxury(image_path):
    '''
    Classify the smartphone brand and predict the intensity of luxury level from the given image.

    :param image_path: str, the path to the image file
    :return: dict, the predicted smartphone brand and luxury level
    '''
    # Load the image classification model
    model_name = 'laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg'
    image_classifier = pipeline('image-classification', model=model_name)

    # Define the smartphone brands and luxury levels
    class_names = [
        'Apple', 'Samsung', 'Huawei', 'Xiaomi',
        'low luxury level', 'medium luxury level', 'high luxury level'
    ]

    # Get the prediction
    result = image_classifier(image_path, class_names)
    return result


# test_function_code --------------------

def test_classify_smartphone_brand_and_luxury():
    print("Testing classify_smartphone_brand_and_luxury function.")

    # Assume 'test_image.jpg' is a valid image path in test dataset
    test_image_path = 'test_image.jpg'

    # Test case: Classifying a known smartphone image
    print("Test case: Known smartphone image classification")
    expected_brand = 'Apple'  # Assuming the known brand for this test
    expected_luxury_level = 'high luxury level'  # Assuming the known luxury level for this test
    result = classify_smartphone_brand_and_luxury(test_image_path)

    # Verify the result contains correct predictions
    assert any(expected_brand in prediction['name'] for prediction in result), "Failed to classify smartphone brand correctly."
    assert any(expected_luxury_level in prediction['name'] for prediction in result), "Failed to classify luxury level correctly."

    print("All tests passed!")

# Run the test
test_classify_smartphone_brand_and_luxury()
