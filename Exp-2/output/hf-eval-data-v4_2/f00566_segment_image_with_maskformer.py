# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_image_with_maskformer(image_url):
    """Segment objects in an image using MaskFormer model.

    Args:
        image_url (str): URL of the image to be segmented.

    Returns:
        dict: A dictionary containing the segmented image map along with other details.

    Raises:
        ValueError: If the image URL is invalid.
    """
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-tiny-coco')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-tiny-coco')

    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError(f'Invalid image URL: {e}')

    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    return result

# test_function_code --------------------

def test_segment_image_with_maskformer():
    print("Testing started.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'  # Sample image URL

    # Test case 1: Valid image URL
    print("Testing case [1/1] started.")
    result = segment_image_with_maskformer(image_url)
    assert 'segmentation' in result, f"Test case [1/1] failed: 'segmentation' key not found in result."
    print("Testing finished.")

# call_test_function_line --------------------

test_segment_image_with_maskformer()