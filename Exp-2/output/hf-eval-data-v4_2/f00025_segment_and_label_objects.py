# requirements_file --------------------

!pip install -U transformers Pillow requests

# function_import --------------------

from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_and_label_objects(image_url):
    """
    Segments and labels objects in an image using a pre-trained MaskFormer model.

    Args:
        image_url (str): The URL of the image to be processed.

    Returns:
        Image: The image with labeled objects.

    Raises:
        ValueError: If the image cannot be loaded from the URL.
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError(f'Unable to load image from URL: {e}')

    processor = MaskFormerImageProcessor.from_pretrained('facebook/maskformer-swin-large-ade')
    inputs = processor(images=image, return_tensors='pt')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-large-ade')
    outputs = model(**inputs)
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    return predicted_semantic_map

# test_function_code --------------------

def test_segment_and_label_objects():
    print("Testing started.")
    image_url = 'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg'

    # Testing case 1: Check if function returns an `Image` object
    print("Testing case [1/1] started.")
    result = segment_and_label_objects(image_url)
    assert isinstance(result, Image.Image), f"Test case [1/1] failed: Expected result to be an instance of Image, got {type(result)} instead."
    print("Testing finished.")

# call_test_function_line --------------------

test_segment_and_label_objects()