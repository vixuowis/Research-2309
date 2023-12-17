# requirements_file --------------------

!pip install -U torch Pillow transformers

# function_import --------------------

import io
import torch
from PIL import Image
from transformers import DetrForSegmentation, DetrFeatureExtractor

# function_code --------------------

def process_image_for_segmentation(image_path):
    """
    Process an image and return the segmented output using DETR model.

    Args:
        image_path (str): The path to the image file to be processed.

    Returns:
        Image: PIL Image object with the segmented objects displayed.

    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If the image processing fails.
    """

    # Load the image
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'Image file does not exist at the path: {image_path}')

    # Initialize the feature extractor and model from pre-trained
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
    model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')

    # Prepare the image for the model
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Perform inference
    outputs = model(**inputs)

    # Extract segmented objects in image
    try:
        results = feature_extractor.post_process_panoptic(outputs, inputs['pixel_values'].shape[-2:], threshold=0.85)
    except Exception as e:
        raise Exception(f'Failed to process image for segmentation: {e}')

    # Convert results to image
    panoptic_seg = results[0]['png_string']
    segmented_image = Image.open(io.BytesIO(panoptic_seg))

    # Return the segmented image
    return segmented_image


# test_function_code --------------------

import io
import tempfile
from PIL import Image

def test_process_image_for_segmentation():
    print("Testing started.")

    # Create a temporary image file for testing
    image = Image.new('RGB', (100, 100), color = (73, 109, 137))
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

        # Testing case 1: Valid image path
        print("Testing case [1/1] started.")
        try:
            result_image = process_image_for_segmentation(tmp_path)
            assert result_image is not None, f"Test case [1/1] failed: Expected a PIL Image object, got {type(result_image)}"
        except Exception as e:
            assert False, f"Test case [1/1] failed with exception: {e}"

    print("Testing finished.")


# call_test_function_line --------------------

test_process_image_for_segmentation()