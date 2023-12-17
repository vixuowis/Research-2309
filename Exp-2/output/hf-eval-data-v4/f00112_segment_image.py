# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_image(image_url: str):
    """
    Segment the provided image using Segformer model.

    Args:
    image_url (str): URL of the image to be segmented.

    Returns:
    torch.Tensor: The logits representing the segmented image.
    """
    # Load the feature extractor and model with pretrained weights
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640')

    # Open the image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Extract features
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Segment the image
    outputs = model(**inputs)

    # Obtain logits
    logits = outputs.logits

    return logits

# test_function_code --------------------

def test_segment_image():
    print("Testing started.")

    # Sample image URL (it would be better to use a local stable image or mock the requests in real tests)
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Call the segment_image function
    logits = segment_image(image_url)

    # Tests for the function output
    assert logits is not None, "Test case failed: The function did not return any logits."
    assert logits.shape[-2:] == (640, 640), f"Test case failed: Expected logits size (640, 640), got {logits.shape[-2:]}"
    print("Testing finished.")

# Run tests
if __name__ == '__main__':
    test_segment_image()