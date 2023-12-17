# requirements_file --------------------

!pip install -U torch, numpy, transformers, Pillow, requests, io

# function_import --------------------

import io
import requests
from PIL import Image
import torch
from transformers import DetrForSegmentation, DetrFeatureExtractor
from transformers.models.detr.feature_extraction_detr import rgb_to_id

# function_code --------------------

def segment_drone_images(image_url):
    """
    Segments objects in an image captured by a drone using the DETR model.

    Parameters:
    image_url (str): The URL of the image to segment.

    Returns:
    numpy.ndarray: The segmented image as a numpy array with objects identified by unique IDs.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
    model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    processed_sizes = torch.as_tensor(inputs['pixel_values'].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
    panoptic_seg_id = rgb_to_id(panoptic_seg)
    return panoptic_seg_id

# test_function_code --------------------

def test_segment_drone_images():
    print("Testing segment_drone_images function.")
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Test case 1: Segmenting an image
    print("Testing case [1/1] started.")
    segmented_result = segment_drone_images(test_url)
    assert segmented_result is not None, "Test case [1/1] failed: The segmented_result should not be None."
    assert isinstance(segmented_result, numpy.ndarray), "Test case [1/1] failed: The result should be a numpy array."
    assert segmented_result.max() > 0, "Test case [1/1] failed: There should be at least one object identified."
    print("Testing finished.")

# Run the test function
test_segment_drone_images()