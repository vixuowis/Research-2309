# requirements_file --------------------

!pip install -U transformers pillow requests

# function_import --------------------

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# function_code --------------------

def perform_image_segmentation(image_path):
    """
    Performs semantic segmentation on an image using a pre-trained Segformer model.

    Parameters:
    image_path (str): The path to the image file or a URL to the image.

    Returns:
    dict: A dictionary containing the segmented output.
    """
    # Load the feature extractor and model
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')

    # Load image
    if image_path.startswith('http'):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)

    # Prepare the inputs
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Perform segmentation
    outputs = model(**inputs)
    logits = outputs.logits

    # Convert the logits to segmentation map
    segmentation_map = logits.argmax(dim=1)[0]

    # Get segmented output as a dictionary
    segmented_output = {'segmentation_map': segmentation_map.numpy()}
    return segmented_output

# test_function_code --------------------

def test_perform_image_segmentation():
    print("Testing perform_image_segmentation function.")
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    result = perform_image_segmentation(image_url)

    # Test case: Check if the result contains the correct key
    assert 'segmentation_map' in result, "Test case failed: 'segmentation_map' key not found in the result."

    # Test case: Check if the segmentation map is not empty
    assert result['segmentation_map'].size > 0, "Test case failed: The segmentation map is empty."

    print("Testing perform_image_segmentation function completed successfully.")

# Run the test function
test_perform_image_segmentation()