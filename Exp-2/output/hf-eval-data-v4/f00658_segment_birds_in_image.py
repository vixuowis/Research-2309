# requirements_file --------------------

!pip install -U torch, transformers, PIL, requests

# function_import --------------------

from PIL import Image
import requests
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# function_code --------------------

def segment_birds_in_image(image_url):
    """
    Segment birds in a provided image using Mask2Former model.

    Parameters:
    image_url (str): URL of the image to perform segmentation on.

    Returns:
    tuple: A tuple containing the original Image object and the predicted instance map (segmentation).
    """
    # Initialize the processor and model
    processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-tiny-coco-instance')
    model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-tiny-coco-instance')

    # Load the image
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocess the image
    inputs = processor(images=image, return_tensors='pt')

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the segmentation outputs
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_instance_map = result['segmentation']

    return image, predicted_instance_map

# test_function_code --------------------

def test_segment_birds_in_image():
    print("Testing segmentation function.")
    test_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'  # Image with birds from COCO dataset

    original_image, segmentation_map = segment_birds_in_image(test_url)

    assert original_image is not None, 'Failed to load the image for segmentation.'
    assert segmentation_map is not None, 'Failed to perform segmentation on the image.'
    print("Test passed successfully!")

# Run the test
test_segment_birds_in_image()