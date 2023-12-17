# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_bean_disease(image_path: str):
    """
    Detects bean diseases by classifying an image of a bean crop leaf.

    Args:
        image_path (str): The file path to the image of a bean crop leaf.

    Returns:
        dict: The classification result including the disease name and confidence score.

    Raises:
        FileNotFoundError: If the image file is not found at the specified path.
        Exception: If there is an error during the image classification process.
    """
    classifier = pipeline('image-classification', model='fxmarty/resnet-tiny-beans')
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'{image_path} does not exist.')
    try:
        results = classifier(image_path)
        return results[0]
    except Exception as e:
        raise


# test_function_code --------------------

def test_detect_bean_disease():
    print("Testing started.")
    sample_image_path = 'test_image.jpg'  # Replace with an appropriate test image path

    # Test case 1: Valid image path
    print("Testing case [1/2] started.")
    try:
        result = detect_bean_disease(sample_image_path)
        assert isinstance(result, dict), f"Test case [1/2] failed: Expected result to be a dict but got {type(result)}"
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"

    # Test case 2: Invalid image path
    print("Testing case [2/2] started.")
    try:
        detect_bean_disease('non_existent_image.jpg')
        assert False, "Test case [2/2] failed: FileNotFoundError not raised as expected."
    except FileNotFoundError:
        pass  # Expected exception
    except Exception as e:
        assert False, f"Test case [2/2] failed: Expected FileNotFoundError, but got {e}"
    print("Testing finished.")


# call_test_function_line --------------------

test_detect_bean_disease()