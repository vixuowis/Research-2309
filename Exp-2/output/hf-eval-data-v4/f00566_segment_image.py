# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_image(url):
    """
    Segment the image from the given URL by identifying objects and drawing
    boundaries around them using the MaskFormer model.

    Parameters:
    - url (str): URL of the image to process.

    Returns:
    - dict: A dictionary containing the 'segmentation' with recognized objects
      and their boundaries.
    """

    # Load pre-trained feature extractor and model
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-tiny-coco')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-tiny-coco')

    # Open the image from the URL
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Get the object detection results and segmentation masks
    outputs = model(**inputs)

    # Post-process segmentation results
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    # Extract predicted panoptic segmentation map
    predicted_panoptic_map = result['segmentation']

    return predicted_panoptic_map

# test_function_code --------------------

def test_segment_image():
    print("Testing segment_image function.")

    # Test URL of an image
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Obtain the segmentation result
    segmentation_map = segment_image(test_url)

    # Test case: Check if the output is a dictionary with 'segmentation' key
    print("Testing if the output contains 'segmentation' key.")
    assert isinstance(segmentation_map, dict) and 'segmentation' in segmentation_map, "The output must be a dictionary with a 'segmentation' key."

    print("All tests passed for segment_image function.")

# Run the test function
test_segment_image()