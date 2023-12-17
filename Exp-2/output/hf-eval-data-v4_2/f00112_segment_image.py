# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_image(image_url: str) -> 'Image':
    """
    Process and segment an image using the Segformer model for semantic segmentation.

    Args:
        image_url: A string URL pointing to the image to be segmented.

    Returns:
        A PIL Image object representing the segmented image.

    Raises:
        ValueError: If the input URL does not point to a valid image.
        RuntimeError: If the model fails to process or segment the image.
    """
    # Load feature extractor and segmentation model
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')

    # Load the image from the given URL
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise ValueError("Invalid image URL or unable to retrieve the image.") from e

    # Extract features from the image
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Perform segmentation on the image
    try:
        outputs = model(**inputs)
        logits = outputs.logits
        segmented_image = feature_extractor.decode_segmentation(logits.squeeze(0))
    except Exception as e:
        raise RuntimeError("Model failed to process the image.") from e

    return segmented_image

# test_function_code --------------------

def test_segment_image():
    print("Testing started.")
    
    # Test case 1: Verify an image can be segmented correctly
    print("Testing case [1/1] started.")
    try:
        segmented_image = segment_image('http://images.cocodataset.org/val2017/000000039769.jpg')
        assert isinstance(segmented_image, Image.Image), f"Test case [1/1] failed: The output is not an instance of PIL.Image.Image"
    except Exception as e:
        assert False, f"Test case [1/1] failed with exception: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_segment_image()