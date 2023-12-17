# requirements_file --------------------

import subprocess

requirements = ["transformers", "pillow", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_urban_scene(image_url: str) -> 'Image.Image':
    """
    Segments an urban scene image using a pretrained Segformer model.

    Args:
        image_url: A string that represents the URL of the image to be processed.

    Returns:
        An Image.Image object with the segmented urban scene.

    Raises:
        ValueError: If the image_url is empty or not reachable.

    """
    # Validate the image URL
    if not image_url:
        raise ValueError('The image URL must not be empty.')

    # Initialize feature extractor and model
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b2-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-cityscapes-1024-1024')

    # Fetch the image from the URL and convert to RGB
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert('RGB')
    except requests.exceptions.RequestException as e:
        raise ValueError(f'Failed to fetch image from URL: {e}')

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Perform segmentation
    outputs = model(**inputs)
    # Convert logits to pixel values
    seg_image = feature_extractor.decode_segmentation(outputs.logits.argmax(dim=1).squeeze(0).cpu().numpy())

    # Convert pillow image
    segmented_image = Image.fromarray(seg_image)
    return segmented_image

# test_function_code --------------------

def test_segment_urban_scene():
    print("Testing started.")
    valid_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    invalid_image_url = ''
    unreachable_image_url = 'http://example.com/non_existing_image.jpg'

    # Test case 1: Valid URL
    print("Testing case [1/3] started.")
    try:
        result = segment_urban_scene(valid_image_url)
        assert isinstance(result, Image.Image), f"Test case [1/3] failed: Result is not an image instance."
    except Exception as e:
        assert False, f"Test case [1/3] failed: {e}"

    # Test case 2: Empty URL
    print("Testing case [2/3] started.")
    try:
        segment_urban_scene(invalid_image_url)
        assert False, "Test case [2/3] failed: ValueError not raised for empty URL."
    except ValueError:
        assert True
    except Exception as e:
        assert False, f"Test case [2/3] failed: {e}"

    # Test case 3: Unreachable URL
    print("Testing case [3/3] started.")
    try:
        segment_urban_scene(unreachable_image_url)
        assert False, "Test case [3/3] failed: ValueError not raised for unreachable URL."
    except ValueError:
        assert True
    except Exception as e:
        assert False, f"Test case [3/3] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_segment_urban_scene()