# requirements_file --------------------

!pip install -U transformers PIL requests

# function_import --------------------

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

# function_code --------------------

def segment_aerial_images(image_url):
    """
    Segment the objects in the aerial images using MaskFormer model from Hugging Face.

    Parameters:
        image_url (str): The URL of the aerial image to be processed.

    Returns:
        dict: A dictionary containing the segmented map of the image.
    """
    # Load the image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Initialize the feature extractor and the model
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained('facebook/maskformer-swin-base-ade')
    model = MaskFormerForInstanceSegmentation.from_pretrained('facebook/maskformer-swin-base-ade')

    # Prepare the inputs for the model
    inputs = feature_extractor(images=image, return_tensors='pt')

    # Perform the segmentation
    outputs = model(**inputs)

    # Post-process the outputs to obtain the segmentation map
    predicted_semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    return predicted_semantic_map

# test_function_code --------------------

def test_segment_aerial_images():
    print("Testing started.")
    test_image_url = 'https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg'  # Example image URL

    # Test case 1: Check if the function returns a dictionary
    print("Testing case [1/1] started.")
    result = segment_aerial_images(test_image_url)
    assert isinstance(result, dict), f"Test case [1/1] failed: Expected a dictionary, got {type(result)}"
    print("Testing finished.")

# Run the test function
test_segment_aerial_images()