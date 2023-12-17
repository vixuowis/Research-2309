# requirements_file --------------------

!pip install -U transformers>=4.11.0

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_car_damage(image_path):
    # Loads the pre-trained zero-shot classification model
    classifier = pipeline('image-classification', model='laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    class_names = ['major accident', 'minor damages']
    # Classify the damage based on the image provided
    result = classifier(image_path, class_names)
    return result

# test_function_code --------------------

def test_classify_car_damage():
    print("Testing started.")
    image_path = 'path_to_test_image.jpg' # Replace with path to test image

    # Expected result should be one of the class_names
    expected_results = ['major accident', 'minor damages']
    result = classify_car_damage(image_path)
    assert result[0]['label'] in expected_results, f"Test failed: Result {result[0]['label']} not in {expected_results}"
    print("Test passed.")

# Run the test function
test_classify_car_damage()