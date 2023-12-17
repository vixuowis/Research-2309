# requirements_file --------------------

import subprocess

requirements = ["transformers", "pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests

# function_code --------------------

def apply_image_segmentation(image_url: str) -> dict:
    """
    Applies image segmentation to the given image URL using OneFormerForUniversalSegmentation model.

    Args:
        image_url: A string with the URL of the image to be segmented.

    Returns:
        A dictionary containing the 'semantic_map' with the segmented output.
    """
    # Load the image from the provided URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Load the OneFormer model and processor
    processor = OneFormerProcessor.from_pretrained('shi-labs/oneformer_coco_swin_large')
    model = OneFormerForUniversalSegmentation.from_pretrained('shi-labs/oneformer_coco_swin_large')

    # Prepare the inputs for the model
    semantic_inputs = processor(images=image, task_inputs=['semantic'], return_tensors='pt')

    # Apply segmentation and obtain the output
    semantic_outputs = model(**semantic_inputs)
    predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]

    # Return the segmented image map
    return {'semantic_map': predicted_semantic_map}

# test_function_code --------------------

def test_apply_image_segmentation():
    print("Testing started.")
    test_image_url = 'https://huggingface.co/datasets/shi-labs/oneformer_demo/blob/main/coco.jpeg'  # A valid image URL for testing

    # Test case 1: Check if the function returns a dictionary
    print("Testing case [1/1] started.")
    result = apply_image_segmentation(test_image_url)
    assert isinstance(result, dict), f"Test case [1/1] failed: Expected result type is dict, got {type(result).__name__}."
    print("Testing finished.")

# call_test_function_line --------------------

test_apply_image_segmentation()