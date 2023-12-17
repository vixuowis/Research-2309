# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def estimate_distance(image):
    """
    Estimate the depth map of a given construction site image using the pretrained depth estimation model.

    :param image: An image of a construction site.
    :return: A depth map indicating the estimated distances.
    """
    # Instantiate the model using the specified depth-estimation pipeline
    depth_model = pipeline('depth-estimation', model='sayakpaul/glpn-nyu-finetuned-diode')

    # Generate the depth map
    depth_map = depth_model(image)
    return depth_map

# test_function_code --------------------

def test_estimate_distance():
    # Import necessary libraries for testing
    import numpy as np
    from PIL import Image

    print("Testing estimate_distance function.")
    # Load a sample image
    test_image_path = 'sample_construction_site.jpg'
    test_image = Image.open(test_image_path)

    # Call the distance estimation function
    estimated_depth_map = estimate_distance(test_image)

    # Check if the result is not None
    assert estimated_depth_map is not None, 'The depth map should not be None.'
    print("Test passed successfully.")

# Run the test
test_estimate_distance()