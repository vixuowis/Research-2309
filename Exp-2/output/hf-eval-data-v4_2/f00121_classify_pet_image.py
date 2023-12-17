# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_pet_image(image_path: str, labels: list[str]) -> dict:
    """
    Classifies an image based on the given labels using a zero-shot classification model.

    Args:
        image_path (str): The file path to the image to classify.
        labels (list[str]): A list of strings representing the possible class labels.

    Returns:
        dict: A dictionary with keys 'labels' and 'scores' containing the predicted label and the associated confidence score.

    Raises:
        FileNotFoundError: If the image file does not exist at the specified path.
        ValueError: If the provided labels list is empty.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image file not found at specified path: {image_path}')
    if not labels:
        raise ValueError('The labels list must not be empty.')

    classifier = pipeline('image-classification', model='laion/CLIP-convnext_base_w-laion2B-s13B-b82K')
    classification_result = classifier(image_path, labels)
    return classification_result


# test_function_code --------------------

def test_classify_pet_image():
    print("Testing started.")
    sample_image_path = 'path/to/sample_image.jpg'  # Assuming a sample image file exists at this location

    # Test case 1: Valid classification with labels 'cat' and 'dog'.
    print("Testing case [1/2] started.")
    result = classify_pet_image(sample_image_path, ['cat', 'dog'])
    assert 'labels' in result and 'scores' in result, "Test case [1/2] failed: classification results should contain 'labels' and 'scores'"

    # Test case 2: Raise FileNotFoundError for non-existent image path.
    print("Testing case [2/2] started.")
    try:
        classify_pet_image('nonexistent/path.jpg', ['cat', 'dog'])
        assert False, "Test case [2/2] failed: FileNotFoundError should be raised for non-existent image path."
    except FileNotFoundError:
        pass
    print("Testing finished.")


# call_test_function_line --------------------

test_classify_pet_image()