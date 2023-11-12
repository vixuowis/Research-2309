# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_aerial_image(image_url):
    """
    Segments an aerial image into different categories such as streets, buildings, and trees.

    Args:
        image_url (str): The URL of the image to be segmented.

    Returns:
        segmentation_map (dict): A dictionary containing the segmented map of the image.

    Raises:
        FileNotFoundError: If the image file does not exist at the specified URL.
    """
    image = Image.open(requests.get(image_url, stream=True).raw)
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_ade20k_swin_large')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_ade20k_swin_large')

    segmentation_inputs = processor(images=image, task_inputs=['semantic'], return_tensors='pt')
    segmentation_outputs = model(**segmentation_inputs)
    segmentation_map = processor.post_process_semantic_segmentation(segmentation_outputs, target_sizes=[image.size[::-1]])[0]
    return segmentation_map

# test_function_code --------------------

def test_segment_aerial_image():
    """
    Tests the segment_aerial_image function with different test cases.
    """
    test_image_url = 'https://placekitten.com/200/300'
    try:
        segmentation_map = segment_aerial_image(test_image_url)
        assert isinstance(segmentation_map, dict), 'The segmentation map should be a dictionary.'
    except FileNotFoundError:
        print('Test image not found.')
    else:
        print('All Tests Passed')

# call_test_function_code --------------------

test_segment_aerial_image()