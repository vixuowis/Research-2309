# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_image(image_path):
    """
    Segment an image using a pretrained Segformer model.

    Args:
        image_path (str): The path to the image file that needs to be segmented.

    Returns:
        tuple: A tuple containing:
                   -a PIL Image of the segmented output
                   -the raw logits from the model
    """
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    image = Image.open(requests.get(image_path, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    segmented_image = logits.argmax(1)[0]
    return image, segmented_image

# test_function_code --------------------

def test_segment_image():
    print("Testing 'segment_image' function.")
    test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    original_image, segmented_image = segment_image(test_image_url)

    assert original_image is not None, "Test failed: The original image is not loaded correctly."
    assert segmented_image.size() == original_image.size, "Test failed: The segmented image dimensions do not match the original image dimensions."

    print("Test for 'segment_image' function passed.")