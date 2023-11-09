# function_import --------------------

from some_module import UperNetModel

# function_code --------------------

def perform_image_segmentation(image):
    """
    This function performs semantic segmentation on an input image using a pre-trained UperNet model.

    Args:
        image (PIL.Image or np.array): The input image to be segmented. It can be either a PIL Image object or a numpy array.

    Returns:
        np.array: The segmented image. Each pixel in the image is assigned a label that corresponds to a particular object class.

    Raises:
        ValueError: If the input image is not a PIL Image object or a numpy array.
    """
    # Load the pre-trained UperNet model
    model = UperNetModel.from_pretrained('openmmlab/upernet-convnext-small')

    # Perform semantic segmentation on the input image
    segmented_image = model(image)

    return segmented_image

# test_function_code --------------------

def test_perform_image_segmentation():
    """
    This function tests the 'perform_image_segmentation' function by using a sample image.
    """
    # Load a sample image
    image = load_sample_image()

    # Perform semantic segmentation on the sample image
    segmented_image = perform_image_segmentation(image)

    # Check if the output is a numpy array
    assert isinstance(segmented_image, np.array), 'The output should be a numpy array.'

    # Check if the output image has the same shape as the input image
    assert segmented_image.shape == image.shape, 'The output image should have the same shape as the input image.'

# call_test_function_code --------------------

test_perform_image_segmentation()