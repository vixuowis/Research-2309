# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def estimate_depth(image_path):
    """
    Estimates the depth map of the given image using the glpn-nyu-finetuned-diode model.

    Args:
        image_path (str): The file path to the image for which depth needs to be estimated.

    Returns:
        dict: A dictionary containing the depth map.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If the image is not in the correct format.

    """
    import os
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist.")

    try:
        # Load and preprocess the image
        depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode')
        depth_map = depth_estimator(image_path)
        return depth_map
    except Exception as e:
        raise ValueError(f"Invalid image format or pipeline error: {e}")

# test_function_code --------------------

def test_estimate_depth():
    print("Testing started.")
    # Load a sample image from a predefined file path
    image_path = 'sample_image.jpg'

    # Test case 1: Image file does not exist
    print("Testing case [1/3] started.")
    non_existing_image = 'non_existing_image.jpg'
    try:
        estimate_depth(non_existing_image)
    except FileNotFoundError as e:
        assert str(e) == f"{non_existing_image} does not exist.", f"Test case [1/3] failed: {e}"

    # Test case 2: Invalid image format
    print("Testing case [2/3] started.")
    invalid_image = 'invalid_image.txt'
    try:
        estimate_depth(invalid_image)
    except ValueError as e:
        assert 'Invalid image format or pipeline error' in str(e), f"Test case [2/3] failed: {e}"

    # Test case 3: Valid image
    print("Testing case [3/3] started.")
    try:
        result = estimate_depth(image_path)
        assert isinstance(result, dict), "Test case [3/3] failed: Depth map is not a dictionary"
    except Exception as e:
        assert False, f"Test case [3/3] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_estimate_depth()