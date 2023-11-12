# function_import --------------------

from transformers import UperNetModel
from PIL import Image
import numpy as np

# function_code --------------------

def perform_image_segmentation(image_path):
    """
    Perform semantic segmentation on an image using a pre-trained UperNet model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.array: Segmented image array.

    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If any error occurs during the segmentation process.
    """
    try:
        # Load the pre-trained UperNet model
        model = UperNetModel.from_pretrained('openmmlab/upernet-convnext-small')

        # Load the image
        image = Image.open(image_path)

        # Perform segmentation
        segmented_image = model(image)

        return segmented_image
    except FileNotFoundError as fnf_error:
        print(f'Error: {fnf_error}')
        raise
    except Exception as e:
        print(f'Error: {e}')
        raise

# test_function_code --------------------

def test_perform_image_segmentation():
    """
    Test the perform_image_segmentation function.
    """
    # Test with a valid image path
    try:
        segmented_image = perform_image_segmentation('test_image.jpg')
        assert isinstance(segmented_image, np.array)
        print('Test case 1 passed')
    except Exception as e:
        print(f'Test case 1 failed: {e}')

    # Test with an invalid image path
    try:
        segmented_image = perform_image_segmentation('invalid_path.jpg')
    except FileNotFoundError:
        print('Test case 2 passed')
    except Exception as e:
        print(f'Test case 2 failed: {e}')

# call_test_function_code --------------------

test_perform_image_segmentation()