# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_satellite_image(image_url):
    """
    Segments a satellite image and returns the semantic map of the image.

    Args:
        image_url (str): The URL of the satellite image to be segmented.

    Returns:
        dict: A dictionary containing the segmented semantic map of the image.

    Raises:
        Exception: If there is an error in downloading the image or processing the segmentation.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)

        processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_coco_swin_large')
        model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_coco_swin_large')

        semantic_inputs = processor(images=image, task_inputs=['semantic'], return_tensors='pt')
        semantic_outputs = model(**semantic_inputs)

        predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]
        return predicted_semantic_map
    except Exception as e:
        raise Exception(f'An error occurred while segmenting the image: {e}')

# test_function_code --------------------

def test_segment_satellite_image():
    print("Testing started.")
    image_url = 'https://huggingface.co/datasets/shi-labs/oneformer_demo/blob/main/coco.jpeg'  # Replace with an appropriate satellite image URL

    # Test case 1: Check if the function returns a dictionary.
    print("Testing case [1/1] started.")
    semantic_map = segment_satellite_image(image_url)
    assert isinstance(semantic_map, dict), f"Test case [1/1] failed: Expected a dictionary, got {type(semantic_map)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_segment_satellite_image()