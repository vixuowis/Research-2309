# requirements_file --------------------

!pip install -U transformers pytorch datasets tokenizers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_diabetic_retinopathy(image_path):
    """
    Uses a pretrained model to classify an image for the presence of diabetic retinopathy.
    
    Parameters:
    - image_path: str - Path to the image that needs to be classified.
    
    Returns:
    - dict: Classification result returned by the model.
    """
    # Load the image classification model
    image_classifier = pipeline('image-classification', 'martinezomg/vit-base-patch16-224-diabetic-retinopathy')
    # Classify the given image
    result = image_classifier(image_path)
    return result

# test_function_code --------------------

def test_classify_diabetic_retinopathy():
    print("Testing started.")
    # Assuming a valid image path for testing and expecting a specific result format.
    test_image_path = "test_eye.jpg"
    expected_result_keys = ['label', 'score']

    # Testing case: Function returns a dictionary with keys 'label' and 'score'
    print("Testing case [1/1] started.")
    result = classify_diabetic_retinopathy(test_image_path)
    assert isinstance(result, list) and all(isinstance(res, dict) and all(key in res for key in expected_result_keys) for res in result), \
        f"Test case [1/1] failed: The result should be a list of dictionaries with keys {expected_result_keys}."
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_classify_diabetic_retinopathy()