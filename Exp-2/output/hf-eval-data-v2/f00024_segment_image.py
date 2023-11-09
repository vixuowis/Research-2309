# function_import --------------------

from transformers import DetrForSegmentation, DetrFeatureExtractor
from PIL import Image
import torch

# function_code --------------------

def segment_image(image_path):
    """
    This function segments an image using the pre-trained 'facebook/detr-resnet-50-panoptic' model.

    Args:
        image_path (str): The path to the image to be segmented.

    Returns:
        A dictionary containing the segmented image and the processed sizes.
    """
    # Load the image
    image = Image.open(image_path)

    # Load the feature extractor and model
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
    model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')

    # Prepare the inputs
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Get the outputs from the model
    outputs = model(**inputs)

    # Process the sizes
    processed_sizes = torch.as_tensor(inputs['pixel_values'].shape[-2:]).unsqueeze(0)

    # Post process the outputs
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

    return result

# test_function_code --------------------

def test_segment_image():
    """
    This function tests the 'segment_image' function.
    """
    # Define the image path
    image_path = 'test_image.jpg'

    # Call the 'segment_image' function
    result = segment_image(image_path)

    # Assert that the result is a dictionary
    assert isinstance(result, dict)

    # Assert that the dictionary has the expected keys
    assert 'png_string' in result and 'segments_info' in result

# call_test_function_code --------------------

test_segment_image()