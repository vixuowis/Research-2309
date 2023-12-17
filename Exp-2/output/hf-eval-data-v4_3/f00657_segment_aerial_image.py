# requirements_file --------------------

import subprocess

requirements = ["transformers", "PIL", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_aerial_image(image_url: str) -> Image.Image:
    """
    Segments an aerial photograph of a city into different classes such as streets, buildings, and trees.

    Args:
        image_url (str): The URL of the aerial image to be segmented.

    Returns:
        Image.Image: An image object with the segmentation map.

    Raises:
        IOError: If the image cannot be opened from the given URL.
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
    print("Testing started.")
    image_url = 'https://huggingface.co/datasets/shi-labs/oneformer_demo/blob/main/ade20k.jpeg'

    # Test case 1: Check if function returns an Image object
    print("Testing case [1/1] started.")
    result = segment_aerial_image(image_url)
    assert isinstance(result, Image.Image), f"Test case [1/1] failed: Expected result to be an instance of Image.Image, got {type(result)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_segment_aerial_image()