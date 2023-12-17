# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def estimate_depth_from_image(image_path):
    """
    Estimate the depth map of an image captured by a drone.

    Args:
        image_path (str): The file path to the image.

    Returns:
        numpy.ndarray: An array containing the depth map.
    """
    # Initialize the depth estimation model
    depth_estimator = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode')
    # Load the image and estimate the depth map
    with open(image_path, 'rb') as image_file:
        depth_map = depth_estimator(image_file)

    return depth_map

# test_function_code --------------------

def test_estimate_depth_from_image():
    print("Testing estimate_depth_from_image function.")
    sample_image_path = 'sample_image.jpg'  # Assuming we have a sample image in the same directory
    depth_map = estimate_depth_from_image(sample_image_path)

    # Test case: Check if the depth map is not None
    assert depth_map is not None, "Depth map is None, depth estimation failed."
    print("Testing estimate_depth_from_image function completed successfully.")

# Run the test
if __name__ == '__main__':
    test_estimate_depth_from_image()